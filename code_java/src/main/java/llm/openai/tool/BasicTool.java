package llm.openai.tool;


import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class BasicTool implements ToolInterface{

  @Tool(name = "get_current_time", description = "Get the current time in a specific timezone")
  public String getCurrentTime(
      @ToolParam(
          name = "timezone", description = "Timezone name, e.g., 'Asia/Tokyo', 'America/New_York', 'Europe/London'")
      String timezone) {
    try {
      ZoneId zoneId = ZoneId.of(timezone);
      LocalDateTime now = LocalDateTime.now(zoneId);
      DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
      return String.format("Current time in %s: %s", timezone, now.format(formatter));
    } catch (Exception e) {
      return "Error: Invalid timezone. Try 'Asia/Tokyo' or 'America/New_York'";
    }
  }

  @Tool(name = "schedule_management_create", description = "日程创建")
  public String scheduleManagementCreate(
      @ToolParam(name = "name", description = "日程名称") String name,
      @ToolParam(name = "description", description = "日程描述") String description,
      @ToolParam(name = "time", description = "日程时间, 格式为 yyyy-MM-dd hh:mm:ss") String time
      ) {
    log.info("创建日程: {}, 描述: {}, 时间: {}", name, description, time);
    return "日程 " + name + " 创建成功，描述: " + description + "，时间: " + time;
  }
}
