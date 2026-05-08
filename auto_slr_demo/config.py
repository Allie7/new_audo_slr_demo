"""项目配置文件"""
import os
from dotenv import load_dotenv

load_dotenv()

# 系统用 LLM 配置 (GLM)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "1a45c9131af54b2f9c681ed19644ebbd.X3KhahYaxFIxwHZy")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "glm-5.1")

# 模拟学生用 LLM 配置 (DeepSeek)
SIMULATOR_API_KEY = os.getenv("SIMULATOR_API_KEY", "sk-f0a04027c57a45f0ba018f4312bc3135")
SIMULATOR_BASE_URL = os.getenv("SIMULATOR_BASE_URL", "https://api.deepseek.com/")
SIMULATOR_MODEL = os.getenv("SIMULATOR_MODEL", "deepseek-chat")

# 数据存储目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Semantic Scholar API Key（可选，不设则使用匿名访问，速率较低）
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "TFI7OTccgm4AkXmmvL3G3zZu7nAl5XR8YfgKlxQj")

# 代理配置（可选，用于访问 Semantic Scholar / arXiv 等海外 API）
# 示例: http://127.0.0.1:7890 或 socks5://127.0.0.1:1080
HTTP_PROXY = os.getenv("HTTP_PROXY", "")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")

# 论文搜索配置
MAX_PAPERS_TO_SEARCH = 100        # 每次搜索返回的最大论文数
SAMPLE_INCLUDE_THRESHOLD = 3      # 初步筛选样例的 include 阈值
MAX_SATISFIED_PAPERS = 5         # 达标论文最大数量
CONTRACTOR_CONTEXT_ROUNDS = 3     # contactor 记住的历史轮数
