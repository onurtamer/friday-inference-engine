// Byte-pair encoding must apply merges by rank, not greedy longest match.
// Vocab {a, b, c, ab, bc} with merges ranked [b c] < [a b]: BPE("abc") = [a, bc],
// while greedy longest match gives [ab, c].
// KNOWN FAILING until the tokenizer applies merges (roadmap R1).
#include <fstream>
#include <string>
#include <vector>
#include "check.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
    const std::string path = (argc > 1) ? argv[1] : "tokenizer_bpe_fixture.json";
    {
        std::ofstream f(path);
        f << R"({"unk_token_id": 5, "eos_token_id": 6,
  "id_to_token": {"0":"a","1":"b","2":"c","3":"ab","4":"bc","5":"<unk>","6":"<eos>"},
  "token_to_id": {"a":0,"b":1,"c":2,"ab":3,"bc":4,"<unk>":5,"<eos>":6},
  "merges": ["b c", "a b"]})";
    }
    Tokenizer tok(path);
    const std::vector<int> ids = tok.encode("abc");
    const std::vector<int> expected = {0, 4};
    std::fprintf(stderr, "encode(\"abc\") =");
    for (int id : ids) std::fprintf(stderr, " %d", id);
    std::fprintf(stderr, " (expected 0 4)\n");
    CHECK(ids == expected);
    return CHECK_RESULT();
}
