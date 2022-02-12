#ifndef __FEC_DECODER__
#define __FEC_DECODER__

#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include <glog/logging.h>

#include "fecpp.h"

namespace relay {
namespace fec {

constexpr size_t packet_size = 512;

struct IncomingFrame {
  size_t index;
  size_t n;
  size_t k;
  size_t size;
  std::map<size_t, const uint8_t*> pool;
  std::vector<size_t> index_pool;

  IncomingFrame(size_t fream_index, size_t frame_n, size_t frame_k,
                size_t frame_size) {
    index = fream_index;
    n = frame_n;
    k = frame_k;
    size = frame_size;
  }
};

class FECDecoder {
 public:
  void reset();

  // i: index, n: n, k: k, r: received share, s: frame size
  bool push(size_t i, size_t n, size_t k, size_t r, size_t s, char* ptr);

  std::string& getOutput();
  bool hasOutput() const;

 protected:
  static constexpr size_t reset_threshold_ = 10;
  static constexpr size_t buffer_size_ = 64;

  char buffer[buffer_size_][packet_size];

  std::unordered_set<size_t> available_buffer_;
  std::vector<IncomingFrame> incoming_frames_;

  size_t index_ = 0;
  size_t old_package_received_ = 0;
  bool prev_old_package_ = false;

  std::string output_;
  bool has_output_ = false;
};

}  // namespace fec
}  // namespace relay

#endif
