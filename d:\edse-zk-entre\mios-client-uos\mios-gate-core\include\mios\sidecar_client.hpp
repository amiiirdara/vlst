#pragma once

#include "mios/config.hpp"

#include <string>

namespace mios {

struct SidecarResult {
  bool ok = false;
  std::string message;
};

class SidecarClient {
 public:
  explicit SidecarClient(ClientConfig config);

  SidecarResult health() const;
  SidecarResult connect_server() const;
  SidecarResult sync_offline() const;
  SidecarResult bind_client(const std::string& client_id,
                            const std::string& client_name) const;

 private:
  ClientConfig config_;
  std::string http_post_json(const std::string& path, const std::string& body) const;
};

}  // namespace mios
