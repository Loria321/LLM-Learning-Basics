import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===================== 爬取配置 =====================
# 替换为你的Cookie（已确认生效）
BAIDU_COOKIE = "BAIDUID_BFESS=19F5BE2708F8675C04D98E7492F422C7:FG=1; BDUSS=RUY2tacU0xeWJ5M2pZc3BySGg1aGRJblpsek04WThjaFBoRjl4dXJ4VH5oRDVvSVFBQUFBJCQAAAAAAQAAAAEAAAAiReJ-wuTTotPr0-DqzQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP~3Fmj~9xZoTj; BDUSS_BFESS=RUY2tacU0xeWJ5M2pZc3BySGg1aGRJblpsek04WThjaFBoRjl4dXJ4VH5oRDVvSVFBQUFBJCQAAAAAAQAAAAEAAAAiReJ-wuTTotPr0-DqzQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP~3Fmj~9xZoTj; PSTM=1764841894; BIDUPSID=4AF2D4D4A67E7D3118BD2A9C8D30262C; H_WISE_SIDS_BFESS=63140_65314_66109_66213_66189_66226_66275_66262_66393_66516_66529_66561_66584_66594_66600_66562_66611_66655_66681_66666_66692_66714_66717_66743_66787_66791_66804_66799_66599_66816; __bid_n=19ae8c6f4d57536b0c6311; H_PS_PSSID=63140_65314_66226_66275_66393_66529_66561_66584_66594_66600_66655_66681_66666_66692_66714_66717_66743_66787_66791_66804_66799_66849_66599_66606_66882; H_WISE_SIDS=63140_65314_66226_66275_66393_66529_66561_66584_66594_66600_66655_66681_66666_66692_66714_66717_66743_66787_66791_66804_66799_66849_66599_66606_66882; STOKEN=8d469ef6cb9827a22d7ec4986e52c06a08694450067e25be63f3705503e37b42; BAIDU_WISE_UID=wapp_1767444706215_966; USER_JUMP=-1; Hm_lvt_292b2e1608b0823c1cb6beef7243ef34=1767444708; HMACCOUNT=2D5C9163DA9366D5; BAIDU_SSP_lcr=https://www.quark.cn/s/J23SQ2wUoMQlhkmqsd?from=kkframenew_resultsearch&uc_param_str=ntnwvepffrbiprsvchutosstxs&by=submit&q=%E8%B4%B4%E5%90%A7&queryId=RGpqM2HRYaskow3a7liinIpNcV3vLZo59ArhzLZm1a6EVpBeWtTrblY7JHEnf46fsToROqcxI8p6ZqJBG8cAq21vc16bY; st_key_id=17; arialoadData=false; video_bubble6423725346=1; __itrace_wid=609ba60f-5142-4bd7-0e2f-cfb6675296ed; wise_device=0; ZFY=KqTy6dBLe:AFADMIIcQtaCDRLg9Kf6KHe3tUpp:B5tIRw:C; 6423725346_FRSVideoUploadTip=1; TIEBA_SID=H4sIAAAAAAAAA9MFAPiz3ZcBAAAA; XFI=1b6b2ff0-e8a6-11f0-8122-7fce924fe4e3; XFCS=262D129461094A1C10D63300243055F8F9B2E057E5473DF20C6AC4AF5FCB99AE; XFT=3HsvRForPs5qsW9C/CfKXyuqPUAyz2XFadzxlObLh9Q=; Hm_lpvt_292b2e1608b0823c1cb6beef7243ef34=1767446227; BA_HECTOR=0020ak8h04ak8kal80050hal8l0l831kli5mj26; ariaappid=c890648bf4dd00d05eb9751dd0548c30; ariauseGraymode=false; ab_sr=1.0.1_OTZlMGUzYzRiMWFhOTdmZTAxM2EyZWZlNzIwZjE4MzI4NDA2MzMyY2ViNzc3NWZhMDA5NjAxOGY0MGU4MWZiOGFlZWE0MTU5Mzg5YzkzNmMyZTVkNzYyYWFiYTQyZTU3YjQzYzIyZDRiMjA5MDg1Y2E1ZWQ0MGNjODU2YmUyMjJlMzQ2OWZlOTIxNWZlZDAwOWI1YWRiNjg3ZjdjNTcxOTRlZGEyYmYyNmIyNGZiZWJiZmJiMzk5MGIwZjg1OWE3; st_data=2bf448974b05b8cb428c6675eca06f7d98513cba8fb3df8dfbe4c36b86e6d223d72bad90688ccb646142edef9e1f5f9399284bf8427d3f8778591fdd6d1702769581003578d2e95ea6333b73f51ce8359e69fc2ef88a0bba8b9fa3ae94dbef63dd4dfa4c1babe1fcb1db799ad987869b783752cfa2d1792d2e8d27d89e397f465684343fb4647e2853c095925edd8234; st_sign=649fbcb4"  
# 你的Cookie
# 杭电吧首页URL
BASE_URL = "https://tieba.baidu.com/f?kw=%E6%9D%AD%E5%B7%9E%E7%94%B5%E5%AD%90%E7%A7%91%E6%8A%80%E5%A4%A7%E5%AD%A6&ie=utf-8"
# 先关闭关键词筛选（爬取所有帖子，后续再筛选）
QUESTION_KEYWORDS = []  # 改为空列表，先爬所有帖子
PAGE_COUNT = 2  # 仍爬2页测试
# 完整请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": BASE_URL,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Cookie": BAIDU_COOKIE
}
OUTPUT_CSV = "hdutieba_qa_raw.csv"

# ===================== 工具函数（适配最新结构） =====================
def create_session():
    """创建带重试的会话"""
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        backoff_factor=2,
        status_forcelist=[403, 500, 503]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session

def get_page_soup(session, url):
    """获取页面并解析，增加调试打印"""
    try:
        time.sleep(random.uniform(2, 5))
        response = session.get(url, timeout=15)
        response.encoding = "utf-8"
        if response.status_code == 200:
            # 调试：打印页面前500字符，确认页面正常加载
            print(f"🔍 页面加载成功，前500字符：{response.text[:500]}")
            return BeautifulSoup(response.text, "html.parser")
        else:
            print(f"⚠️  状态码：{response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️  请求异常：{str(e)}")
        return None

def extract_post_links(session, soup):
    """适配最新贴吧结构：提取所有帖子链接（兼容新旧class）"""
    post_links = []
    # 方案1：适配最新class（2026贴吧帖子标题class）
    post_items = soup.find_all("a", class_="thread-title-abs")
    # 方案2：兼容旧class（兜底）
    if not post_items:
        post_items = soup.find_all("a", class_="j_th_tit")
    # 方案3：通过href筛选（终极兜底）
    if not post_items:
        post_items = soup.find_all("a", href=lambda x: x and "/p/" in x and "pn=" not in x)
    
    print(f"🔍 找到 {len(post_items)} 个帖子项（页面解析结果）")
    for item in post_items:
        post_title = item.get_text(strip=True) if item.get_text else ""
        # 提取链接（兼容不同结构）
        post_href = item.get("href", "")
        if post_href and not post_href.startswith("http"):
            post_href = "https://tieba.baidu.com" + post_href
        if post_title and post_href:
            post_links.append({"title": post_title, "url": post_href})
    
    # 即使无关键词，也打印数量
    print(f"✅ 本页提取到 {len(post_links)} 条帖子（无关键词筛选）")
    return post_links

def extract_post_content(session, post_url):
    """提取帖子正文和回复（适配最新结构）"""
    soup = get_page_soup(session, post_url)
    if not soup:
        return "", []
    
    # 提取正文（兼容新旧结构）
    post_content = ""
    # 最新结构：class="p_content_wrap"
    content_wrap = soup.find("div", class_="p_content_wrap")
    if content_wrap:
        content = content_wrap.find("div", class_="d_post_content")
        if content:
            post_content = content.get_text(strip=True).replace("\n", " ").replace("\t", " ")
    # 旧结构兜底
    if not post_content:
        content_div = soup.find("div", class_="d_post_content_main")
        if content_div:
            content = content_div.find("div", class_="d_post_content j_d_post_content")
            if content:
                post_content = content.get_text(strip=True).replace("\n", " ").replace("\t", " ")
    
    # 提取回复（兼容新旧结构）
    replies = []
    reply_divs = soup.find_all("div", class_=lambda x: x and "l_post" in x)
    for i, reply_div in enumerate(reply_divs[1:4]):  # 跳过楼主
        reply_content = reply_div.find("div", class_=lambda x: x and "d_post_content" in x)
        if reply_content:
            reply_text = reply_content.get_text(strip=True).replace("\n", " ").replace("\t", " ")
            if reply_text and len(reply_text) > 5:
                replies.append(reply_text)
    
    return post_content, replies

# ===================== 核心爬取逻辑 =====================
def crawl_hdutieba():
    print("="*60)
    print("📌 爬取杭电吧数据（适配最新结构+调试模式）")
    print(f"爬取页数：{PAGE_COUNT} | 关键词筛选：{'关闭' if not QUESTION_KEYWORDS else '开启'}")
    print("="*60)
    
    session = create_session()
    qa_data = []
    
    for page in range(PAGE_COUNT):
        print(f"\n📄 正在爬取第 {page+1} 页...")
        # 确认分页参数：贴吧分页是pn=(page+1)*50？测试两种参数
        page_url = f"{BASE_URL}&pn={(page+1)*50}"  # 修正分页参数
        soup = get_page_soup(session, page_url)
        if not soup:
            continue
        
        # 提取帖子链接
        post_links = extract_post_links(session, soup)
        if not post_links:
            print(f"❌ 第 {page+1} 页无有效帖子链接")
            continue
        
        # 爬取帖子详情
        for idx, post in enumerate(post_links):
            print(f"  正在爬取帖子 {idx+1}/{len(post_links)}：{post['title'][:20]}...")
            post_content, replies = extract_post_content(session, post["url"])
            
            # 合并文本
            full_content = f"标题：{post['title']} 正文：{post_content} 回复：{' | '.join(replies)}"
            qa_data.append({
                "title": post["title"],
                "post_url": post["url"],
                "content": full_content
            })
    
    # 保存数据
    if qa_data:
        df = pd.DataFrame(qa_data)
        # 爬取后再筛选含问答关键词的帖子（兜底）
        if QUESTION_KEYWORDS:
            df = df[df["title"].str.contains("|".join(QUESTION_KEYWORDS))]
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"\n✅ 爬取完成！共获取 {len(df)} 条数据")
        print(f"📁 数据已保存至：{OUTPUT_CSV}")
    else:
        print("\n❌ 未获取到有效数据（请检查页面解析逻辑）")

# ===================== 运行 =====================
if __name__ == "__main__":
    crawl_hdutieba()