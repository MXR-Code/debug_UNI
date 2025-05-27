import subprocess

def download_with_wget(url, filename, log_filename):
    """使用wget下载文件，并将输出重定向到日志文件"""
    try:
        command = ["wget", url, "-O", filename]
        with open(log_filename, "w") as log_file:
            subprocess.run(command, stdout=log_file, stderr=log_file, check=True)
        print(f"已成功下载 {filename} (查看日志文件 {log_filename} 获取详细信息)")
    except subprocess.CalledProcessError as e:
        print(f"下载 {filename} 失败: {e}")
    except FileNotFoundError:
        print("wget 命令未找到，请确保已安装 wget。")


train_url = "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K-NONORM.zip?download=1"
test_url = "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1"

download_with_wget(train_url, filename="NCT-CRC-HE-100K-NONORM.zip", log_filename="output.logcrc1")
download_with_wget(test_url, filename="CRC-VAL-HE-7K.zip", log_filename="output.logcrc2")