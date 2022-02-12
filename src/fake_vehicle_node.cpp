#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "udp_send_socket.h"
#include "udp_receive_socket.h"
#include "messages.h"
#include "practical_socket.h"

DEFINE_string(foreign_addr, "127.0.0.1", "Foreign address");
DEFINE_int32(foreign_port, 7000, "Foreign server port");
DEFINE_int32(foreign_from_port, 7001, "Foreign port to receive message from");
DEFINE_int32(foreign_to_port, 7002, "Foreign port to send message to");

std::atomic_bool running;

void myHandler(int s) { running.store(false); }

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = myHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  running.store(true);

  try {
    LOG(INFO) << "Connecting to server: " << FLAGS_foreign_addr << ":"
              << FLAGS_foreign_port;
    int32_t speed = 0;

    TCPSocket sock(FLAGS_foreign_addr, FLAGS_foreign_port);
    LOG(INFO) << "Connected!";

    relay::communication::UDPSendSocket modi_to_sock(8000); // TODO: change this
    modi_to_sock.setForeign(FLAGS_foreign_addr, FLAGS_foreign_to_port);
    relay::communication::UDPReceiveSocket modi_from_sock(FLAGS_foreign_from_port);

    while (running.load() && modi_to_sock.running() && modi_from_sock.running()) {
      modi_from_sock.consume([&](std::deque<std::string>& buffer) {
        if (!buffer.empty()) {
          if (buffer.back().size() == ControlCommand::size) {
            ControlCommand command =
                ControlCommand::parseControlCommand(buffer.back().c_str());
            LOG(INFO) << "Receive <<< ";
            LOG(INFO) << "TR: " << static_cast<int>(command.takeover_request);
            LOG(INFO) << "YC: " << static_cast<int>(command.yaw_control);
            LOG(INFO) << "TC: " << static_cast<int>(command.throttle_control);
            LOG(INFO) << "GC: " << static_cast<int>(command.gear);
          }
          buffer.clear();
        }
      });

      VehicleState state = {0, 0, speed};
      char buffer[VehicleState::size];
      state.makeVehicleState(buffer);
      modi_to_sock.push(VehicleState::size, buffer);
      // LOG(INFO) << "Send <<< ";
      // LOG(INFO) << "CM: " << static_cast<int>(state.control_mode);
      // LOG(INFO) << "GR: " << static_cast<int>(state.gear);
      // LOG(INFO) << "SP: " << static_cast<int>(state.speed);
      speed += 1000;
    }
    modi_to_sock.stop();
    modi_from_sock.stop();
  } catch (SocketException& e) {
    LOG(FATAL) << "Error occurred when accepting connection: " << e.what();
  }
  LOG(INFO) << "Bye bye";

  return 0;
}
