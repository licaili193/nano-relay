#include "fecpp.h"

#include "fec_decoder.h"

namespace relay {
namespace fec {

void FECDecoder::reset() {
  LOG(INFO) << "FEC Decoder reset triggered";

  index_ = 0;
  available_buffer_.clear();
  for (size_t i = 0; i < buffer_size_; i++) {
    available_buffer_.insert(i);
  }
  incoming_frames_.clear();
  prev_old_package_ = false;
  old_package_received_ = 0;

  has_output_ = false;
}

bool FECDecoder::push(size_t i, size_t n, size_t k, size_t r, size_t s,
                      char* ptr) {
  // Step 1: drop if it is behind the index
  if (i < index_) {
    if (prev_old_package_) {
      ++old_package_received_;
      if (old_package_received_ > reset_threshold_) {
        LOG(INFO) << "There is a likely reconnection. Reseting...";
        reset();
        return false;
      }
    } else {
      old_package_received_ = 0;
    }
    prev_old_package_ = true;
    return false;
  }

  // Step 2: check if we have existing incoming frame
  bool has_frame = false;
  size_t frame_index = 0;
  for (size_t j = 0; j < incoming_frames_.size(); j++) {
    if (incoming_frames_[j].index == i) {
      has_frame = true;
      frame_index = j;
      break;
    }
  }

  // Step 3: create the incoming frame if we don't have one
  if (!has_frame) {
    if (n >= buffer_size_ || k >= n || s > packet_size * k) {
      LOG(ERROR) << "Bad package received once - n: " 
                 << n << " k: "
                 << k << " s: "
                 << s;
      return false;
    }
    incoming_frames_.emplace_back(i, n, k, s);
    frame_index = incoming_frames_.size() - 1;
  }
  IncomingFrame& frame = incoming_frames_[frame_index];

  // Step 4: insert packet
  if (available_buffer_.empty()) {
    LOG(ERROR) << "There are too many packages in FEC Decoder";
    reset();
    return false;
  }
  size_t buffer_index = *available_buffer_.begin();
  available_buffer_.erase(buffer_index);
  memcpy(buffer[buffer_index], ptr, packet_size);
  frame.index_pool.push_back(buffer_index);
  frame.pool[r] = reinterpret_cast<uint8_t*>(buffer[buffer_index]);

  // Step 5: check if we have an output
  if (frame.index_pool.size() >= frame.k) {
    uint8_t temp_buffer[packet_size * buffer_size_];
    try {
      fecpp::fec_code codec(frame.k, frame.n);
      codec.decode(frame.pool, packet_size,
                   [&](size_t i, size_t k, const uint8_t* ptr, size_t s) {
                     memcpy(temp_buffer + i * s, ptr, s);
                   });
    } catch (std::logic_error e) {
      LOG(ERROR) << "FEC Decoder failed to decode once";
      return false;
    }
    std::string new_output(reinterpret_cast<char*>(temp_buffer), frame.size);
    output_ = std::move(new_output);
    has_output_ = true;

    index_ = frame.index;
    std::unordered_set<size_t> to_erase;
    for (size_t j = 0; j < incoming_frames_.size(); j++) {
      if (incoming_frames_[j].index <= index_) {
        to_erase.insert(j);
        for (auto idx : incoming_frames_[j].index_pool) {
          available_buffer_.insert(idx);
        }
      }
    }
    std::vector<IncomingFrame> new_incoming_frames;
    for (auto& f : incoming_frames_) {
      if (to_erase.find(f.index) == to_erase.end()) {
        new_incoming_frames.push_back(std::move(f));
      }
    }
    incoming_frames_ = std::move(new_incoming_frames);
  }
}

std::string& FECDecoder::getOutput() {
  has_output_ = false;
  return output_;
}

bool FECDecoder::hasOutput() const { return has_output_; }

}  // namespace fec
}  // namespace relay
