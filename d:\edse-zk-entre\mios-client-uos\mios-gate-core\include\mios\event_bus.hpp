#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace mios {

/** Push IPC event frames to subscribed clients. */
class EventBus {
 public:
  using Subscriber = std::function<void(const std::string& event_line)>;

  int subscribe(Subscriber sub);
  void unsubscribe(int id);
  void publish(const std::string& event_name, const std::string& params_json);

 private:
  std::mutex mutex_;
  int next_id_ = 1;
  std::vector<std::pair<int, Subscriber>> subscribers_;
};

}  // namespace mios
