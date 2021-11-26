#include <string.h>
#include <memory>
#include <iostream>

#include "fecpp.h"

#include "udp_send_socket.h"

namespace relay {
namespace communication {

UDPSendSocket::UDPSendSocket(unsigned short port, 
                             size_t max_buffer_size,
                             std::string foreign_addr,
                             unsigned short foreign_port)
    : sock_(port) {
  max_buffer_size_ = max_buffer_size;

  setForeign(foreign_addr, foreign_port);

  running_.store(true);
  thread_ = std::thread(&UDPSendSocket::worker, this);
}

void UDPSendSocket::setForeign(std::string foreign_addr,
                               unsigned short foreign_port) {
  foreign_addr_ = foreign_addr;
  foreign_port_ = foreign_port;
}

void UDPSendSocket::worker() {
  sock_.setBlocking(false);
  try {
    while (running_.load()) {
      sock_.check();

      std::lock_guard<std::mutex> guard(mutex_);
      while (!buffer_.empty()) {
        const auto& payload = buffer_.front();
        uint8_t packet_num = (payload.size() + packet_size - 1) / packet_size;
        if (!(1 <= packet_num && packet_num + extra_packet <= 256)) {
          if (packet_num) {
            LOG(WARNING) 
                << "Packet number does match FEC requirements. Skiping...";
          }
          buffer_.pop_front();
          continue;
        }
        fecpp::fec_code codec(packet_num, packet_num + extra_packet);
        codec.encode(reinterpret_cast<const uint8_t*>(payload.c_str()),
                     packet_num * packet_size,
                     [&](size_t i, size_t n, const uint8_t* ptr, size_t s) {
                       size_t send_size = std::min(s, packet_size);
                       PacketHeader header = {frame_index_,
                                              packet_num + extra_packet,
                                              i,
                                              packet_num,
                                              payload.size(),
                                              0};
                       header.makePacketHeader(send_buffer_);
                       memcpy(send_buffer_ + header_size, ptr, s);
                       sock_.sendTo(send_buffer_, header_size + packet_size,
                                    foreign_addr_, foreign_port_);
                     });
        frame_index_++;

        buffer_.pop_front();
      }

      // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (SocketException& e) {
    LOG(FATAL) << "Error occurred when sending messages: " << e.what();
  }
}

void UDPSendSocket::stop() { running_.store(false); }

void UDPSendSocket::push(size_t payload_size, const char* payload) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (buffer_.size() >= max_buffer_size_) {
    buffer_.pop_front();
  }
  buffer_.emplace_back(payload, payload_size);
  uint8_t packet_num = (buffer_.back().length() + packet_size - 1) / packet_size;
  buffer_.back().reserve(packet_num * packet_size);
}

void UDPSendSocket::push(std::string payload) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (buffer_.size() >= max_buffer_size_) {
    buffer_.pop_front();
  }
  buffer_.push_back(std::move(payload));
}

bool UDPSendSocket::running() { return running_.load(); }

void UDPSendSocket::join() { thread_.join(); }

void UDPSendSocket::detach() { thread_.detach(); }

}  // namespace communication
}  // namespace relay
