#ifndef __PACKET_HEADER__
#define __PACKET_HEADER__

#include <string.h>

namespace relay {
namespace communication {

constexpr size_t header_size = 19;
constexpr size_t packet_size = 512;
constexpr size_t extra_packet = 2;

struct PacketHeader {
  uint32_t id;
  uint8_t packet_num;
  uint8_t packet_index;
  uint8_t packet_needed;
  // For non-fec application, size means the packet size; for fec application,
  // size means frame size
  uint32_t size;
  uint64_t timestamp;

  void makePacketHeader(char* buffer) {
    memcpy(buffer, &id, 4);
    memcpy(buffer + 4, &packet_num, 1);
    memcpy(buffer + 5, &packet_index, 1);
    memcpy(buffer + 6, &packet_needed, 1);
    memcpy(buffer + 7, &size, 4);
    memcpy(buffer + 11, &timestamp, 8);
  }

  static PacketHeader parsePacketHeader(const char* buffer) {
    PacketHeader res;
    memcpy(&res.id, buffer, 4);
    memcpy(&res.packet_num, buffer + 4, 1);
    memcpy(&res.packet_index, buffer + 5, 1);
    memcpy(&res.packet_needed, buffer + 6, 1);
    memcpy(&res.size, buffer + 7, 4);
    memcpy(&res.timestamp, buffer + 11, 8);
    return res;
  }
};

}  // namespace communication
}  // namespace relay

#endif
