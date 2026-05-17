#pragma once

#include "mios/event_bus.hpp"
#include "mios/gate_runtime.hpp"
#include "mios/sidecar_client.hpp"

#include <string>

namespace mios {

class RpcDispatcher {
 public:
  RpcDispatcher(GateRuntime& runtime, SidecarClient& sidecar, EventBus& events);

  std::string handle_line(const std::string& line);

 private:
  GateRuntime& runtime_;
  SidecarClient& sidecar_;
  EventBus& events_;
};

}  // namespace mios
