package llm.openai.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.openai.core.JsonValue;
import com.openai.models.FunctionDefinition;
import com.openai.models.FunctionParameters;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.Parameter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class ToolBox {
  private static final ObjectMapper ARG_COERCION = new ObjectMapper();

  private List<ToolInterface> tools = new ArrayList<>();

  @Getter
  private List<FunctionDefinition> functionDefinitions = List.of();

  /** function name → 可执行注册信息（实例、Method、与形参顺序一致的 JSON 参数名） */
  private Map<String, ToolRegistration> registrationsByName = Map.of();

  public void addTool(ToolInterface tool) {
    tools.add(tool);
    calcFunctionDefinitions();
  }

  /**
   * 按 {@link FunctionDefinition} 中的 name 调用已注册工具，将 {@code arguments} 按注册时的 JSON 属性名映射到方法形参并反射执行。
   */
  public Object callTool(String name, Map<String, Object> arguments) {
    log.info("callTool name: {}, arguments: {}", name, arguments);
    Objects.requireNonNull(name, "name");
    ToolRegistration reg = registrationsByName.get(name);
    if (reg == null) {
      throw new IllegalArgumentException("Unknown tool: " + name);
    }
    Map<String, Object> args = arguments == null ? Map.of() : arguments;
    Parameter[] parameters = reg.method().getParameters();
    Object[] values = new Object[parameters.length];
    for (int i = 0; i < parameters.length; i++) {
      String key = reg.argumentKeys().get(i);
      Object raw = args.get(key);
      try {
        values[i] = coerceArgument(raw, parameters[i].getType());
      } catch (IllegalArgumentException e) {
        throw new IllegalArgumentException("参数 '" + key + "': " + e.getMessage(), e);
      }
    }
    try {
      return reg.method().invoke(reg.tool(), values);
    } catch (InvocationTargetException e) {
      Throwable c = e.getCause();
      if (c instanceof RuntimeException re) {
        throw re;
      }
      if (c instanceof Error err) {
        throw err;
      }
      throw new RuntimeException(c);
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }

  private void calcFunctionDefinitions() {
    List<FunctionDefinition> defs = new ArrayList<>();
    Map<String, ToolRegistration> byName = new LinkedHashMap<>();
    for (ToolInterface tool : tools) {
      for (Class<?> c = tool.getClass(); c != null && c != Object.class; c = c.getSuperclass()) {
        for (Method method : c.getDeclaredMethods()) {
          if (method.isBridge() || method.isSynthetic()) {
            continue;
          }
          if (!Modifier.isPublic(method.getModifiers())) {
            continue;
          }
          if (!method.isAnnotationPresent(Tool.class)) {
            continue;
          }
          Tool toolAnn = method.getAnnotation(Tool.class);
          String fnName = toolAnn.name().isEmpty() ? method.getName() : toolAnn.name();
          if (byName.containsKey(fnName)) {
            throw new IllegalStateException("重复的 tool 名称: " + fnName);
          }
          List<String> keys = orderedArgumentKeys(method);
          byName.put(fnName, new ToolRegistration(tool, method, keys));
          defs.add(getFunctions(tool, method));
        }
      }
    }
    this.functionDefinitions = List.copyOf(defs);
    this.registrationsByName = Map.copyOf(byName);
  }

  private record ToolRegistration(ToolInterface tool, Method method, List<String> argumentKeys) {}

  private static List<String> orderedArgumentKeys(Method method) {
    List<String> keys = new ArrayList<>(method.getParameterCount());
    for (Parameter parameter : method.getParameters()) {
      keys.add(resolveParamName(parameter, parameter.getAnnotation(ToolParam.class)));
    }
    return keys;
  }

  private static Object coerceArgument(Object raw, Class<?> type) {
    if (raw == null) {
      if (type.isPrimitive()) {
        throw new IllegalArgumentException("不能为 null（基本类型形参）");
      }
      return null;
    }
    if (type.isInstance(raw)) {
      return raw;
    }
    try {
      return ARG_COERCION.convertValue(raw, type);
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException(
          "无法将 " + raw.getClass().getSimpleName() + " 转为 " + type.getName(), e);
    }
  }

  /**
   * 根据方法上的 {@link Tool}、参数上的 {@link ToolParam} 生成 OpenAI {@link FunctionDefinition}（JSON Schema
   * properties / required）。
   *
   * @param object 工具实例（预留，便于后续做实例方法或校验）
   */
  public static FunctionDefinition getFunctions(Object object, Method method) {
    Objects.requireNonNull(object, "tool");
    Objects.requireNonNull(method, "method");

    Tool tool = method.getAnnotation(Tool.class);
    if (tool == null) {
      throw new IllegalArgumentException("方法缺少 @Tool 注解: " + method);
    }
    String name = tool.name().isEmpty() ? method.getName() : tool.name();
    String description = tool.description();

    Map<String, Object> properties = new LinkedHashMap<>();
    List<String> required = new ArrayList<>();

    for (Parameter parameter : method.getParameters()) {
      ToolParam paramAnn = parameter.getAnnotation(ToolParam.class);
      String paramName = resolveParamName(parameter, paramAnn);
      String paramDescription = paramAnn == null ? "" : paramAnn.description();
      if (paramAnn == null || paramAnn.required()) {
        required.add(paramName);
      }
      Map<String, Object> prop = new LinkedHashMap<>();
      prop.put("type", jsonSchemaType(parameter.getType()));
      if (!paramDescription.isEmpty()) {
        prop.put("description", paramDescription);
      }
      properties.put(paramName, prop);
    }

    FunctionParameters.Builder paramsBuilder = FunctionParameters.builder()
        .putAdditionalProperty("type", JsonValue.from("object"))
        .putAdditionalProperty("properties", JsonValue.from(properties));
    if (!required.isEmpty()) {
      paramsBuilder.putAdditionalProperty("required", JsonValue.from(required));
    }

    return FunctionDefinition.builder()
        .name(name)
        .description(description)
        .parameters(paramsBuilder.build())
        .build();
  }

  private static String resolveParamName(Parameter parameter, ToolParam paramAnn) {
    if (paramAnn != null && !paramAnn.name().isEmpty()) {
      return paramAnn.name();
    }
    return parameter.getName();
  }

  private static String jsonSchemaType(Class<?> c) {
    if (c == String.class || c == char.class || c == Character.class) {
      return "string";
    }
    if (c == boolean.class || c == Boolean.class) {
      return "boolean";
    }
    if (c == double.class || c == Double.class || c == float.class || c == Float.class) {
      return "number";
    }
    if (c == byte.class || c == Byte.class
        || c == short.class || c == Short.class
        || c == int.class || c == Integer.class
        || c == long.class || c == Long.class) {
      return "integer";
    }
    return "string";
  }
}
