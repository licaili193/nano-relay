#ifndef __MESSAGE_HEADER__
#define __MESSAGE_HEADER__

#include <stdint.h>
#include <string.h>

struct VehicleState {
  // 0: manual; 1: auto; 2: remote
  uint8_t control_mode;
  // 0: drive; 1: reverse
  uint8_t gear;
  // -max to max: -100 m/s to 100 m/s
  int32_t speed;

  static VehicleState parseVehicleState(const char* buffer) {
    VehicleState res;
    memcpy(&res.control_mode, buffer, 1);
    memcpy(&res.gear, buffer + 1, 1);
    memcpy(&res.speed, buffer + 2, 4);
    return res;
  }

  void makeVehicleState(char* buffer) {
    memcpy(buffer, &control_mode, 1);
    memcpy(buffer + 1, &gear, 1);
    memcpy(buffer + 2, &speed, 4);
  }

  constexpr static size_t size = 6;
};

struct ControlCommand {
  bool takeover_request;
  // Range: -100 to 100, positive direction is right side
  int8_t yaw_control;
  // Range: 0 to 100
  int8_t throttle_control;
  // 0: drive; 1: reverse
  uint8_t gear;

  static ControlCommand parseControlCommand(const char* buffer) {
    ControlCommand res;
    memcpy(&res.takeover_request, buffer, 1);
    memcpy(&res.yaw_control, buffer + 1, 1);
    memcpy(&res.throttle_control, buffer + 2, 1);
    memcpy(&res.gear, buffer + 3, 1);
    return res;
  }

  void makeControlCommand(char* buffer) {
    memcpy(buffer, &takeover_request, 1);
    memcpy(buffer + 1, &yaw_control, 1);
    memcpy(buffer + 2, &throttle_control, 1);
    memcpy(buffer + 3, &gear, 1);
  }

  constexpr static size_t size = 4;
};

#endif
