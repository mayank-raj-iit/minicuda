#!/usr/bin/env python3
"""ptxsim.py - a small PTX checker + simulator for the subset MiniCUDA emits.

Why: lets us test the PTX the compiler writes on a laptop with no NVIDIA GPU.
It does two jobs:
  1. CHECK  - like a (very) small ptxas front end: every register is declared,
              every operand has the right register class for the instruction
              type (e.g. add.s32 needs %r registers, ld.global needs a 64-bit
              address), labels and parameters exist, float immediates use the
              exact 0fXXXXXXXX form.
  2. RUN    - executes a kernel for a whole grid, with shared memory and
              bar.sync, using float32 arithmetic.

Usage:
  ptxsim.py kernel.ptx --check
  ptxsim.py kernel.ptx --kernel saxpy --grid 4 --block 256 --args in.txt --out out.txt
The args/out file formats are the same as `minicuda --run`.
"""
import argparse
import math
import re
import struct
import sys

import numpy as np

F32 = np.float32


class PtxError(Exception):
    pass


# ----------------------------------------------------------------- parsing ---
class Instr:
    __slots__ = ("guard", "neg", "op", "parts", "args", "line", "text")


class Kernel:
    def __init__(self, name):
        self.name = name
        self.params = []          # [(name, type)]
        self.regs = {}            # prefix -> (class, count)
        self.shared = {}          # name -> (offset, size)
        self.shared_size = 0
        self.code = []            # list of Instr
        self.labels = {}          # label -> index into code


def strip_comment(line):
    i = line.find("//")
    return line if i < 0 else line[:i]


def parse(text):
    kernels = {}
    lines = text.split("\n")
    i = 0
    header = {}
    while i < len(lines):
        raw = strip_comment(lines[i]).strip()
        i += 1
        if not raw:
            continue
        if raw.startswith((".version", ".target", ".address_size")):
            k, v = raw.split(None, 1)
            header[k] = v.strip()
            continue
        if raw.startswith(".file"):
            continue
        m = re.match(r"\.visible\s+\.entry\s+([A-Za-z_$][\w$]*)\s*\((.*)$", raw)
        if not m:
            raise PtxError(f"line {i}: unexpected text at module scope: {raw}")
        k = Kernel(m.group(1))
        # parameters until ')'
        rest = m.group(2)
        param_text = rest
        while ")" not in param_text:
            param_text += " " + strip_comment(lines[i]).strip()
            i += 1
        param_text = param_text[: param_text.index(")")]
        for p in [x.strip() for x in param_text.split(",") if x.strip()]:
            pm = re.match(r"\.param\s+\.(u32|s32|f32|u64|b32|b64)\s+([\w$]+)$", p)
            if not pm:
                raise PtxError(f"bad parameter declaration '{p}' in {k.name}")
            k.params.append((pm.group(2), pm.group(1)))
        # find '{'
        while strip_comment(lines[i]).strip() != "{":
            i += 1
        i += 1
        while True:
            if i >= len(lines):
                raise PtxError(f"missing '}}' for kernel {k.name}")
            ln = i + 1
            s = strip_comment(lines[i]).strip()
            i += 1
            if not s:
                continue
            if s == "}":
                break
            if s.startswith(".loc"):
                continue
            rm = re.match(r"\.reg\s+\.(pred|b32|f32|b64)\s+(%[a-z]+)<(\d+)>;$", s)
            if rm:
                k.regs[rm.group(2)] = (rm.group(1), int(rm.group(3)))
                continue
            sm = re.match(r"\.shared\s+\.align\s+(\d+)\s+\.b8\s+([\w$]+)\[(\d+)\];$", s)
            if sm:
                align, name, size = int(sm.group(1)), sm.group(2), int(sm.group(3))
                off = (k.shared_size + align - 1) // align * align
                k.shared[name] = (off, size)
                k.shared_size = off + size
                continue
            lm = re.match(r"([A-Za-z_$][\w$]*):$", s)
            if lm:
                if lm.group(1) in k.labels:
                    raise PtxError(f"line {ln}: duplicate label {lm.group(1)}")
                k.labels[lm.group(1)] = len(k.code)
                continue
            if not s.endswith(";"):
                raise PtxError(f"line {ln}: missing ';': {s}")
            s = s[:-1].strip()
            ins = Instr()
            ins.line, ins.text = ln, s
            ins.guard, ins.neg = None, False
            gm = re.match(r"@(!?)(%p\d+)\s+(.*)$", s)
            if gm:
                ins.neg = gm.group(1) == "!"
                ins.guard = gm.group(2)
                s = gm.group(3)
            pieces = s.split(None, 1)
            ins.op = pieces[0]
            ins.parts = ins.op.split(".")
            ins.args = [a.strip() for a in pieces[1].split(",")] if len(pieces) > 1 else []
            k.code.append(ins)
        kernels[k.name] = k
    for key in (".version", ".target", ".address_size"):
        if key not in header:
            raise PtxError(f"missing {key} directive")
    if header[".address_size"] != "64":
        raise PtxError(".address_size must be 64")
    return kernels


# ---------------------------------------------------------------- checking ---
INT32 = {"s32", "u32", "b32"}
INT64 = {"s64", "u64", "b64"}
SREGS = {f"%{g}.{d}" for g in ("tid", "ntid", "ctaid", "nctaid") for d in "xyz"}


def reg_class(k, name, ins):
    m = re.match(r"(%[a-z]+)(\d+)$", name)
    if not m or m.group(1) not in k.regs:
        raise PtxError(f"line {ins.line}: undeclared register {name} in '{ins.text}'")
    cls, count = k.regs[m.group(1)]
    if int(m.group(2)) >= count:
        raise PtxError(f"line {ins.line}: register {name} out of declared range <{count}>")
    return cls


def want(k, ins, operand, kind):
    """kind: 'b32', 'f32', 'b64', 'pred'.  Registers must match; immediates must fit."""
    if operand.startswith("%"):
        if operand in SREGS:
            raise PtxError(f"line {ins.line}: special register {operand} only allowed in mov.u32")
        cls = reg_class(k, operand, ins)
        if cls != kind:
            raise PtxError(f"line {ins.line}: operand {operand} is .{cls} but instruction needs .{kind}: '{ins.text}'")
        return
    if kind == "f32":
        if not re.match(r"0[fF][0-9A-Fa-f]{8}$", operand):
            raise PtxError(f"line {ins.line}: float immediate must be 0fXXXXXXXX, got '{operand}'")
        return
    if kind == "pred":
        raise PtxError(f"line {ins.line}: predicate operand must be a register: '{ins.text}'")
    if not re.match(r"-?\d+$", operand):
        raise PtxError(f"line {ins.line}: bad integer operand '{operand}' in '{ins.text}'")


def type_kind(t):
    if t in INT32:
        return "b32"
    if t in INT64:
        return "b64"
    if t == "f32":
        return "f32"
    if t == "pred":
        return "pred"
    raise PtxError(f"unsupported type .{t}")


def addr_of(operand, ins):
    m = re.match(r"\[([^\]]+)\]$", operand)
    if not m:
        raise PtxError(f"line {ins.line}: expected [address] operand, got '{operand}'")
    return m.group(1)


def check_kernel(k):
    params = {p for p, _ in k.params}
    for ins in k.code:
        p, a = ins.parts, ins.args
        op = p[0]
        if ins.guard:
            want(k, ins, ins.guard, "pred")
        n_expected = None
        if op in ("add", "sub", "mul", "div", "rem", "min", "max", "and", "or", "xor", "shl", "shr"):
            t = p[-1]
            n_expected = 3
            if op == "mul" and p[1] == "wide":
                want(k, ins, a[0], "b64"); want(k, ins, a[1], "b32"); want(k, ins, a[2], "b32")
            else:
                kind = type_kind(t)
                if op in ("shl", "shr"):
                    want(k, ins, a[0], kind); want(k, ins, a[1], kind); want(k, ins, a[2], "b32")
                else:
                    if a[1][0] != "%" and kind != "pred":
                        raise PtxError(f"line {ins.line}: first source of {ins.op} must be a register")
                    for x in a:
                        want(k, ins, x, kind)
            if kind_needs_rounding(ins) and "rn" not in p:
                raise PtxError(f"line {ins.line}: {ins.op} should specify .rn for exact results")
        elif op in ("mad", "fma"):
            n_expected = 4
            kind = type_kind(p[-1])
            for x in a:
                want(k, ins, x, kind)
        elif op in ("neg", "abs", "not", "sqrt"):
            n_expected = 2
            kind = type_kind(p[-1])
            want(k, ins, a[0], kind); want(k, ins, a[1], kind)
            if a[1][0] != "%":
                raise PtxError(f"line {ins.line}: source of {ins.op} must be a register")
        elif op == "setp":
            n_expected = 3
            kind = type_kind(p[-1])
            want(k, ins, a[0], "pred"); want(k, ins, a[1], kind); want(k, ins, a[2], kind)
            if a[1][0] != "%":
                raise PtxError(f"line {ins.line}: first source of setp must be a register")
        elif op == "selp":
            n_expected = 4
            kind = type_kind(p[-1])
            want(k, ins, a[0], kind); want(k, ins, a[1], kind); want(k, ins, a[2], kind); want(k, ins, a[3], "pred")
        elif op == "cvt":
            n_expected = 2
            dt, st = p[-2], p[-1]
            want(k, ins, a[0], type_kind(dt)); want(k, ins, a[1], type_kind(st))
            if a[1][0] != "%":
                raise PtxError(f"line {ins.line}: cvt source must be a register")
        elif op == "cvta":
            n_expected = 2
            want(k, ins, a[0], "b64"); want(k, ins, a[1], "b64")
        elif op == "mov":
            n_expected = 2
            kind = type_kind(p[-1])
            want(k, ins, a[0], kind)
            if a[1] in SREGS:
                if p[-1] != "u32":
                    raise PtxError(f"line {ins.line}: special registers are read with mov.u32")
            elif a[1] in k.shared:
                if p[-1] not in ("u32", "u64"):
                    raise PtxError(f"line {ins.line}: shared address needs mov.u32/u64")
            else:
                want(k, ins, a[1], kind)
        elif op == "ld":
            n_expected = 2
            space, t = p[1], p[-1]
            want(k, ins, a[0], type_kind(t))
            adr = addr_of(a[1], ins)
            if space == "param":
                if adr not in params:
                    raise PtxError(f"line {ins.line}: unknown parameter {adr}")
                ptype = dict(k.params)[adr]
                if type_kind(ptype) != type_kind(t):
                    raise PtxError(f"line {ins.line}: ld.param type .{t} does not match parameter type .{ptype}")
            elif space == "global":
                want(k, ins, adr, "b64")
            elif space == "shared":
                want(k, ins, adr, "b32")
            else:
                raise PtxError(f"line {ins.line}: unsupported state space {space}")
        elif op == "st":
            n_expected = 2
            space, t = p[1], p[-1]
            adr = addr_of(a[0], ins)
            want(k, ins, adr, "b64" if space == "global" else "b32")
            want(k, ins, a[1], type_kind(t))
        elif op in ("atom", "red"):
            space, t = p[1], p[-1]
            if p[2] != "add":
                raise PtxError(f"line {ins.line}: only .add atomics are supported")
            if op == "atom":
                n_expected = 3
                want(k, ins, a[0], type_kind(t)); adr = addr_of(a[1], ins); val = a[2]
            else:
                n_expected = 2
                adr = addr_of(a[0], ins); val = a[1]
            want(k, ins, adr, "b64" if space == "global" else "b32")
            want(k, ins, val, type_kind(t))
        elif op == "bar":
            n_expected = 1
            if ins.op != "bar.sync" or a != ["0"]:
                raise PtxError(f"line {ins.line}: expected bar.sync 0")
        elif op == "bra":
            n_expected = 1
            if a[0] not in k.labels:
                raise PtxError(f"line {ins.line}: unknown label {a[0]}")
        elif op == "ret":
            n_expected = 0
        else:
            raise PtxError(f"line {ins.line}: unknown instruction '{ins.op}'")
        if n_expected is not None and len(a) != n_expected:
            raise PtxError(f"line {ins.line}: {ins.op} expects {n_expected} operands, got {len(a)}")
        # every register written must be declared too
    if not k.code or k.code[-1].op != "ret" and k.code[-1].op not in ("bra", "bra.uni"):
        pass


def kind_needs_rounding(ins):
    return ins.parts[-1] == "f32" and ins.parts[0] in ("add", "sub", "mul", "div")


# --------------------------------------------------------------- execution ---
def s32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def s64(v):
    v &= 0xFFFFFFFFFFFFFFFF
    return v - (1 << 64) if v & (1 << 63) else v


def f2bits(f):
    return struct.unpack("<I", struct.pack("<f", float(f)))[0]


def bits2f(b):
    return F32(struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0])


class Memory:
    def __init__(self):
        self.bufs = []          # (base, bytearray)
        self.next = 0x10000000

    def alloc(self, nbytes):
        base = self.next
        self.bufs.append((base, bytearray(nbytes)))
        self.next += ((nbytes + 4095) // 4096 + 1) * 4096
        return base

    def find(self, addr, n):
        for base, b in self.bufs:
            if base <= addr and addr + n <= base + len(b):
                return b, addr - base
        raise PtxError(f"out-of-bounds global access at 0x{addr:x}")


class Thread:
    __slots__ = ("tid", "regs", "pc", "done", "at_bar")


def run_kernel(k, mem, grid, block, args):
    if len(args) != len(k.params):
        raise PtxError(f"kernel {k.name} takes {len(k.params)} arguments, got {len(args)}")
    param_vals = {}
    for (pname, ptype), (atype, aval) in zip(k.params, args):
        param_vals[pname] = aval
    code = k.code
    ninstr = 0
    for bz in range(grid[2]):
        for by in range(grid[1]):
            for bx in range(grid[0]):
                ctaid = (bx, by, bz)
                shared = bytearray(k.shared_size)
                threads = []
                for z in range(block[2]):
                    for y in range(block[1]):
                        for x in range(block[0]):
                            t = Thread()
                            t.tid, t.regs, t.pc, t.done, t.at_bar = (x, y, z), {}, 0, False, False
                            threads.append(t)
                while True:
                    for t in threads:
                        if not t.done and not t.at_bar:
                            ninstr += step_thread(k, code, t, mem, shared, ctaid, grid, block, param_vals)
                    if not any(t.at_bar for t in threads):
                        break
                    for t in threads:
                        t.at_bar = False
    return ninstr


def step_thread(k, code, t, mem, shared, ctaid, grid, block, params):
    R = t.regs
    count = 0

    def val(x, kind):
        if x[0] == "%":
            if x not in R:
                return F32(0) if kind == "f32" else (False if kind == "pred" else 0)
            return R[x]
        if kind == "f32":
            return bits2f(int(x[2:], 16))
        return int(x)

    while True:
        count += 1
        if count > 50_000_000:
            raise PtxError("step limit exceeded")
        ins = code[t.pc]
        t.pc += 1
        if ins.guard is not None:
            g = R.get(ins.guard, False)
            if g == ins.neg:
                continue
        p, a, op = ins.parts, ins.args, ins.parts[0]
        T = p[-1]
        if op == "ret":
            t.done = True
            return count
        if op == "bra":
            t.pc = k.labels[a[0]]
            continue
        if op == "bar":
            t.at_bar = True
            return count
        if op == "mov":
            src = a[1]
            if src in SREGS:
                g, d = src[1:].split(".")
                idx = "xyz".index(d)
                v = {"tid": t.tid, "ntid": block, "ctaid": ctaid, "nctaid": grid}[g][idx]
            elif src in k.shared:
                v = k.shared[src][0]
            else:
                v = val(src, type_kind(T))
            R[a[0]] = v
            continue
        if op == "ld":
            space = p[1]
            adr = addr_of(a[1], ins)
            if space == "param":
                R[a[0]] = params[adr]
                continue
            addr = R[adr]
            if space == "global":
                buf, off = mem.find(addr, 4)
            else:
                if addr < 0 or addr + 4 > len(shared):
                    raise PtxError(f"out-of-bounds shared read at {addr}")
                buf, off = shared, addr
            raw = struct.unpack_from("<I", buf, off)[0]
            R[a[0]] = bits2f(raw) if T == "f32" else s32(raw)
            continue
        if op == "st":
            space = p[1]
            addr = R[addr_of(a[0], ins)]
            v = val(a[1], type_kind(T))
            if space == "global":
                buf, off = mem.find(addr, 4)
            else:
                if addr < 0 or addr + 4 > len(shared):
                    raise PtxError(f"out-of-bounds shared write at {addr}")
                buf, off = shared, addr
            struct.pack_into("<I", buf, off, f2bits(v) if T == "f32" else (v & 0xFFFFFFFF))
            continue
        if op in ("atom", "red"):
            space = p[1]
            if op == "atom":
                dst, adr, v = a[0], addr_of(a[1], ins), a[2]
            else:
                dst, adr, v = None, addr_of(a[0], ins), a[1]
            addr = R[adr]
            if space == "global":
                buf, off = mem.find(addr, 4)
            else:
                buf, off = shared, addr
            raw = struct.unpack_from("<I", buf, off)[0]
            if T == "f32":
                old = bits2f(raw)
                new = F32(old + val(v, "f32"))
                struct.pack_into("<I", buf, off, f2bits(new))
            else:
                old = s32(raw)
                struct.pack_into("<I", buf, off, (old + val(v, "b32")) & 0xFFFFFFFF)
            if dst:
                R[dst] = old
            continue
        if op == "cvta":
            R[a[0]] = R[a[1]]
            continue
        if op == "cvt":
            dt, st = p[-2], p[-1]
            x = val(a[1], type_kind(st))
            if dt == "f32" and st == "s32":
                R[a[0]] = F32(x)
            elif dt == "s32" and st == "f32":
                xf = float(x)
                if math.isnan(xf):
                    R[a[0]] = 0
                elif xf >= 2147483648.0:
                    R[a[0]] = 2147483647
                elif xf <= -2147483648.0:
                    R[a[0]] = -2147483648
                else:
                    R[a[0]] = int(xf)  # truncation toward zero (.rzi)
            elif dt == "s64" and st == "s32":
                R[a[0]] = s32(x)
            else:
                raise PtxError(f"unsupported cvt {ins.op}")
            continue
        if op == "setp":
            cmp = p[1]
            kind = type_kind(T)
            x, y = val(a[1], kind), val(a[2], kind)
            if kind == "b32":
                x, y = s32(x), s32(y)
            if cmp == "lt": r = x < y
            elif cmp == "le": r = x <= y
            elif cmp == "gt": r = x > y
            elif cmp == "ge": r = x >= y
            elif cmp == "eq": r = x == y
            elif cmp == "ne": r = x != y
            elif cmp == "neu": r = (x != y) or math.isnan(float(x)) or math.isnan(float(y))
            else: raise PtxError(f"unsupported compare {cmp}")
            R[a[0]] = bool(r)
            continue
        if op == "selp":
            kind = type_kind(T)
            R[a[0]] = val(a[1], kind) if R[a[3]] else val(a[2], kind)
            continue
        if op == "not":
            R[a[0]] = not R[a[1]]
            continue
        kind = type_kind(T)
        if op in ("mad", "fma"):
            x, y, z = (val(v, kind) for v in a[1:4])
            if kind == "f32":
                R[a[0]] = F32(float(x) * float(y) + float(z))  # exact product in double, one rounding
            else:
                R[a[0]] = s32(s32(x) * s32(y) + s32(z))
            continue
        if op in ("neg", "abs", "sqrt"):
            x = val(a[1], kind)
            if kind == "f32":
                R[a[0]] = F32(-x) if op == "neg" else (F32(abs(x)) if op == "abs" else F32(np.sqrt(x)))
            else:
                R[a[0]] = s32(-s32(x)) if op == "neg" else s32(abs(s32(x)))
            continue
        if op == "mul" and p[1] == "wide":
            R[a[0]] = s64(s32(val(a[1], "b32")) * s32(val(a[2], "b32")))
            continue
        x, y = val(a[1], kind), val(a[2], "b32" if op in ("shl", "shr") else kind)
        if kind == "pred":
            R[a[0]] = (x and y) if op == "and" else (x or y) if op == "or" else (x != y)
            continue
        if kind == "f32":
            if op == "add": r = x + y
            elif op == "sub": r = x - y
            elif op == "mul": r = x * y
            elif op == "div":
                with np.errstate(all="ignore"):
                    r = x / y
            elif op == "min": r = np.fmin(x, y)
            elif op == "max": r = np.fmax(x, y)
            else: raise PtxError(f"unsupported f32 op {op}")
            R[a[0]] = F32(r)
            continue
        wrap = s64 if kind == "b64" else s32
        x, y = wrap(x), (s32(y) if op in ("shl", "shr") else wrap(y))
        if op == "add": r = x + y
        elif op == "sub": r = x - y
        elif op == "mul": r = x * y
        elif op == "div":
            if y == 0: raise PtxError("integer division by zero")
            r = abs(x) // abs(y) * (1 if (x >= 0) == (y >= 0) else -1)
        elif op == "rem":
            if y == 0: raise PtxError("integer remainder by zero")
            r = x - (abs(x) // abs(y) * (1 if (x >= 0) == (y >= 0) else -1)) * y
        elif op == "and": r = x & y
        elif op == "or": r = x | y
        elif op == "xor": r = x ^ y
        elif op == "shl": r = 0 if (y & 0xFFFFFFFF) >= 32 else x << y
        elif op == "shr": r = (-1 if x < 0 else 0) if (y & 0xFFFFFFFF) >= 32 else x >> y
        elif op == "min": r = min(x, y)
        elif op == "max": r = max(x, y)
        else: raise PtxError(f"unsupported op {ins.op}")
        R[a[0]] = wrap(r)


# ------------------------------------------------------------------ I/O ---
def load_args(path, mem):
    toks = []
    for line in open(path):
        toks += line.split("#")[0].split()
    args, bufs, i = [], [], 0
    while i < len(toks):
        kind = toks[i]
        if kind == "i32":
            args.append(("i32", int(toks[i + 1]))); i += 2
        elif kind == "f32":
            args.append(("f32", F32(float(toks[i + 1])))); i += 2
        elif kind in ("buf", "zeros"):
            ty, n = toks[i + 1], int(toks[i + 2])
            base = mem.alloc(4 * n)
            buf, _ = mem.find(base, 4 * n)
            i += 3
            if kind == "buf":
                for j in range(n):
                    v = toks[i + j]
                    raw = f2bits(F32(float(v))) if ty == "f32" else (int(v) & 0xFFFFFFFF)
                    struct.pack_into("<I", buf, 4 * j, raw)
                i += n
            bufs.append((len(args), ty, n, base))
            args.append(("ptr", base))
        else:
            raise PtxError(f"bad args token {kind}")
    return args, bufs


def write_bufs(out, mem, bufs):
    for idx, ty, n, base in bufs:
        out.write(f"buf {idx} {ty} {n}\n")
        buf, off = mem.find(base, 4 * n)
        vals = []
        for j in range(n):
            raw = struct.unpack_from("<I", buf, off + 4 * j)[0]
            vals.append("%.9g" % float(bits2f(raw)) if ty == "f32" else str(s32(raw)))
        for j in range(0, n, 8):
            out.write(" ".join(vals[j:j + 8]) + "\n")


def dims(s):
    v = [int(x) for x in s.split(",")]
    return tuple(v + [1] * (3 - len(v)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ptx")
    ap.add_argument("--check", action="store_true", help="only validate the PTX")
    ap.add_argument("--kernel")
    ap.add_argument("--grid", default="1")
    ap.add_argument("--block", default="1")
    ap.add_argument("--args")
    ap.add_argument("--out")
    ap.add_argument("--count", action="store_true", help="print the number of executed instructions")
    o = ap.parse_args()
    try:
        kernels = parse(open(o.ptx).read())
        for k in kernels.values():
            check_kernel(k)
        if o.check:
            print(f"OK: {len(kernels)} kernel(s) passed checks")
            return 0
        k = kernels[o.kernel]
        mem = Memory()
        args, bufs = load_args(o.args, mem)
        n = run_kernel(k, mem, dims(o.grid), dims(o.block), [(t, v) for t, v in args])
        if o.count:
            print(f"executed {n} instructions", file=sys.stderr)
        out = open(o.out, "w") if o.out else sys.stdout
        write_bufs(out, mem, bufs)
        return 0
    except PtxError as e:
        print(f"ptxsim: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
