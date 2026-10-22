// lexer.cpp - reads characters one by one and groups them into tokens.
#include "lexer.hpp"

#include <cctype>
#include <unordered_map>

#include "diag.hpp"

namespace {

const std::unordered_map<std::string, Tok> kKeywords = {
    {"__global__", Tok::KwGlobal}, {"__shared__", Tok::KwShared},
    {"extern", Tok::KwExtern},     {"void", Tok::KwVoid},
    {"int", Tok::KwInt},           {"float", Tok::KwFloat},
    {"if", Tok::KwIf},             {"else", Tok::KwElse},
    {"for", Tok::KwFor},           {"while", Tok::KwWhile},
    {"return", Tok::KwReturn},     {"break", Tok::KwBreak},
    {"continue", Tok::KwContinue},
};

bool isIdentStart(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
bool isIdentChar(char c) { return std::isalnum((unsigned char)c) || c == '_'; }
bool isDigit(char c) { return std::isdigit((unsigned char)c) != 0; }

class Lexer {
 public:
  explicit Lexer(const std::string& s) : src_(s) {}

  std::vector<Token> run() {
    std::vector<Token> out;
    while (true) {
      skipSpaceAndComments();
      if (pos_ >= src_.size()) {
        out.push_back({Tok::End, "<end of file>", line_, col_});
        return out;
      }
      out.push_back(next());
    }
  }

 private:
  const std::string& src_;
  size_t pos_ = 0;
  int line_ = 1, col_ = 1;

  char peek(size_t ahead = 0) const {
    return pos_ + ahead < src_.size() ? src_[pos_ + ahead] : '\0';
  }

  char advance() {
    char c = src_[pos_++];
    if (c == '\n') { line_++; col_ = 1; } else { col_++; }
    return c;
  }

  // True when only spaces/tabs sit between the last newline and pos_.
  bool atLineStart() const {
    for (size_t i = pos_; i > 0; --i) {
      char c = src_[i - 1];
      if (c == '\n') return true;
      if (c != ' ' && c != '\t') return false;
    }
    return true;
  }

  void skipSpaceAndComments() {
    while (pos_ < src_.size()) {
      char c = peek();
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        advance();
      } else if (c == '/' && peek(1) == '/') {          // // line comment
        while (pos_ < src_.size() && peek() != '\n') advance();
      } else if (c == '/' && peek(1) == '*') {          // /* block comment */
        int l = line_, cl = col_;
        advance(); advance();
        while (!(peek() == '*' && peek(1) == '/')) {
          if (pos_ >= src_.size()) throw CompileError(l, cl, "unterminated /* comment");
          advance();
        }
        advance(); advance();
      } else if (c == '#' && atLineStart()) {           // #include ... is ignored
        while (pos_ < src_.size() && peek() != '\n') advance();
      } else {
        return;
      }
    }
  }

  Token next() {
    const int l = line_, c = col_;
    const size_t start = pos_;
    auto make = [&](Tok k) { return Token{k, src_.substr(start, pos_ - start), l, c}; };
    const char ch = peek();

    // identifiers and keywords
    if (isIdentStart(ch)) {
      while (isIdentChar(peek())) advance();
      std::string word = src_.substr(start, pos_ - start);
      auto it = kKeywords.find(word);
      return Token{it != kKeywords.end() ? it->second : Tok::Ident, word, l, c};
    }

    // numbers: 42, 3.5, .5, 1e-3, 2.0f
    if (isDigit(ch) || (ch == '.' && isDigit(peek(1)))) {
      bool isFloat = false;
      while (isDigit(peek())) advance();
      if (peek() == '.') {
        isFloat = true;
        advance();
        while (isDigit(peek())) advance();
      }
      if (peek() == 'e' || peek() == 'E') {
        isFloat = true;
        advance();
        if (peek() == '+' || peek() == '-') advance();
        if (!isDigit(peek())) throw CompileError(l, c, "malformed exponent in number");
        while (isDigit(peek())) advance();
      }
      if (peek() == 'f' || peek() == 'F') { isFloat = true; advance(); }
      if (isIdentChar(peek())) throw CompileError(l, c, "invalid suffix on number");
      return make(isFloat ? Tok::FloatLit : Tok::IntLit);
    }

    // string literal (only used for extern "C")
    if (ch == '"') {
      advance();
      while (peek() != '"') {
        if (pos_ >= src_.size() || peek() == '\n') throw CompileError(l, c, "unterminated string");
        advance();
      }
      advance();
      return make(Tok::StrLit);
    }

    // operators and punctuation: try two-character ones first ("<=" before "<")
    advance();
    auto two = [&](char second, Tok ifTwo, Tok ifOne) {
      if (peek() == second) { advance(); return make(ifTwo); }
      return make(ifOne);
    };
    switch (ch) {
      case '(': return make(Tok::LParen);
      case ')': return make(Tok::RParen);
      case '{': return make(Tok::LBrace);
      case '}': return make(Tok::RBrace);
      case '[': return make(Tok::LBracket);
      case ']': return make(Tok::RBracket);
      case ',': return make(Tok::Comma);
      case ';': return make(Tok::Semi);
      case '.': return make(Tok::Dot);
      case '%': return make(Tok::Percent);
      case '^': return make(Tok::Caret);
      case '*': return two('=', Tok::StarAssign, Tok::Star);
      case '/': return two('=', Tok::SlashAssign, Tok::Slash);
      case '=': return two('=', Tok::EqEq, Tok::Assign);
      case '!': return two('=', Tok::NotEq, Tok::Bang);
      case '&': return two('&', Tok::AndAnd, Tok::Amp);
      case '|': return two('|', Tok::OrOr, Tok::Pipe);
      case '+':
        if (peek() == '+') { advance(); return make(Tok::PlusPlus); }
        return two('=', Tok::PlusAssign, Tok::Plus);
      case '-':
        if (peek() == '-') { advance(); return make(Tok::MinusMinus); }
        return two('=', Tok::MinusAssign, Tok::Minus);
      case '<':
        if (peek() == '<') { advance(); return make(Tok::ShiftL); }
        return two('=', Tok::LessEq, Tok::Less);
      case '>':
        if (peek() == '>') { advance(); return make(Tok::ShiftR); }
        return two('=', Tok::GreaterEq, Tok::Greater);
      default: break;
    }
    throw CompileError(l, c, std::string("unexpected character '") + ch + "'");
  }
};

}  // namespace

std::vector<Token> tokenize(const std::string& source) { return Lexer(source).run(); }
