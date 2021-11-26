#ifndef __UDP_SEND_SOCKET__
#define __UDP_SEND_SOCKET__

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include <glog/logging.h>
#include "practical_socket.h"

#include "packet_header.h"

namespace relay {
namespace communication {

class UDPSendSocket {
 public:
  UDPSendSocket(unsigned short port, 
                size_t max_buffer_size = 10,
                std::string foreign_addr = "127.0.0.1",
                unsigned short foreign_port = 6000);

  void stop();
  void push(size_t payload_size, const char* payload);
  void push(std::string payload);
  bool running();

  void join();
  void detach();

  void setForeign(std::string foreign_addr, unsigned short foreign_port);

 protected:
  std::atomic_bool running_;
  std::mutex mutex_;
  size_t max_buffer_size_ = 10;
  size_t frame_index_ = 0;

  std::thread thread_;

  std::deque<std::string> buffer_;
  char send_buffer_[header_size + packet_size + 1];

  UDPSocket sock_;
  std::string foreign_addr_ = "localhost";
  unsigned short foreign_port_ = 6000;

  void worker();
};

}  // namespace communication
}  // namespace relay

#endif
