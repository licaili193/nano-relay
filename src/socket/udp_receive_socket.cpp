#include <string.h>
#include <memory>

#include "udp_receive_socket.h"

namespace relay {
namespace communication {

UDPReceiveSocket::UDPReceiveSocket(unsigned short port, size_t max_buffer_size)
    : sock_(port) {
  max_buffer_size_ = max_buffer_size;

  received_time_ = std::chrono::system_clock::now();

  running_.store(true);
  thread_ = std::thread(&UDPReceiveSocket::worker, this);
}

void UDPReceiveSocket::worker() {
  char buffer[packet_size + header_size];

  sock_.setBlocking(false);
  try {
    while (running_.load()) {
      sock_.check();

      int recv_sz = 0;
      int accum_recv_sz = 0;
      do {
        std::string foreign_addr;
        unsigned short foreign_port;
        recv_sz = sock_.recvFrom(buffer + accum_recv_sz,
                                 packet_size + header_size - accum_recv_sz,
                                 foreign_addr, foreign_port);
        auto now = std::chrono::system_clock::now();
        if (recv_sz > 0) {
          received_time_ = now;
          accum_recv_sz += recv_sz;
        } else {
          if (now - received_time_ > std::chrono::milliseconds(timeout_ms)) {
            break;
          }
        }
      } while (accum_recv_sz < packet_size + header_size);

      if (accum_recv_sz < packet_size + header_size) {
        continue;
      }

      auto header = PacketHeader::parsePacketHeader(buffer);
      decoder_.push(header.id, 
                    header.packet_num, 
                    header.packet_needed, 
                    header.packet_index, 
                    header.size,
                    buffer + header_size);
      if (decoder_.hasOutput()) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (buffer_.size() >= max_buffer_size_) {
          buffer_.pop_front();
        }
        buffer_.push_back(std::move(decoder_.getOutput()));
      }
    }
  } catch (SocketException& e) {
    LOG(FATAL) << "Error occurred when receiving or sending messages: "
               << e.what();
  }
}

void UDPReceiveSocket::stop() { running_.store(false); }

void UDPReceiveSocket::consume(function<void(std::deque<std::string>&)> fun) {
  std::lock_guard<std::mutex> guard(mutex_);
  fun(buffer_);
}

bool UDPReceiveSocket::running() { return running_.load(); }

void UDPReceiveSocket::join() { thread_.join(); }

void UDPReceiveSocket::detach() { thread_.detach(); }

}  // namespace communication
}  // namespace relay
