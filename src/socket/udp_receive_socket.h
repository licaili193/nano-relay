#ifndef __UDP_RECEIVE_SOCKET__
#define __UDP_RECEIVE_SOCKET__

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include <glog/logging.h>
#include "fec_decoder.h"
#include "practical_socket.h"

#include "packet_header.h"

namespace relay {
namespace communication {

class UDPReceiveSocket {
 public:
  UDPReceiveSocket(unsigned short port, size_t max_buffer_size = 10);

  void stop();
  void consume(std::function<void(std::deque<std::string>&)> fun);
  bool running();

  void join();
  void detach();

 protected:
  std::atomic_bool running_;
  std::mutex mutex_;
  size_t max_buffer_size_ = 10;

  uint32_t index_ = 0;
  std::thread thread_;

  std::chrono::system_clock::time_point received_time_;
  const int timeout_ms = 1000;

  std::deque<std::string> buffer_;

  UDPSocket sock_;
  fec::FECDecoder decoder_;

  void worker();
};

}  // namespace communication
}  // namespace relay

#endif
