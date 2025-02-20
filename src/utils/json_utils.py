import json


def convert_json_array_to_jsonl(input_file, output_file):
    """
    将完整的 JSON 数组文件转换为 JSON Lines 格式
    :param input_file: 输入文件路径（原始 JSON 数组）
    :param output_file: 输出文件路径（转换后的 JSON Lines）
    """
    try:
        # 读取原始 JSON 数组
        with open(input_file, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)

        # 验证输入数据是否为列表
        if not isinstance(data, list):
            raise ValueError("输入文件必须是 JSON 数组格式")

        # 写入 JSON Lines 格式
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for item in data:
                # 将每个对象单独写入一行
                json_line = json.dumps(item, ensure_ascii=False)
                f_out.write(json_line + '\n')

        print(f"转换成功！输出文件: {output_file}")

    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {str(e)}")
    except FileNotFoundError:
        print(f"文件不存在: {input_file}")
    except Exception as e:
        print(f"未知错误: {str(e)}")


if __name__ == "__main__":
    # 使用方法
    input_path = "../../dataset/medician_json/test.json"  # 你的原始 JSON 数组文件
    output_path = "../../dataset/medician_json/test.jsonl"  # 生成的 JSON Lines 文件
    convert_json_array_to_jsonl(input_path, output_path)