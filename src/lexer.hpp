// lexer.hpp - Phase 1: turn source text into a list of tokens.
#pragma once
#include <string>
#include <vector>

// Every kind of "word" the language has.
enum class Tok {
  End, Ident, IntLit, FloatLit, StrLit,
  // keywords
  KwGlobal, KwShared, KwExtern, KwVoid, KwInt, KwFloat,
  KwIf, KwElse, KwFor, KwWhile, KwReturn, KwBreak, KwContinue,
  // punctuation
  LParen, RParen, LBrace, RBrace, LBracket, RBracket, Comma, Semi, Dot,
  // operators
  Plus, Minus, Star, Slash, Percent,
  Assign, PlusAssign, MinusAssign, StarAssign, SlashAssign,
  PlusPlus, MinusMinus,
  EqEq, NotEq, Less, LessEq, Greater, GreaterEq,
  AndAnd, OrOr, Bang, Amp, Pipe, Caret, ShiftL, ShiftR,
};

struct Token {
  Tok kind;
  std::string text;  // the exact characters from the source, e.g. "saxpy" or "+="
  int line = 0, col = 0;
};

// Throws CompileError on a bad character, bad number or unterminated comment.
std::vector<Token> tokenize(const std::string& source);
