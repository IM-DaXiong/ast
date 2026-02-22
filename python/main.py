# -*- coding: utf-8 -*-
import os
import json
import shutil
import re
import datetime
import torch
from tqdm import tqdm
import numpy as np
import soundfile as sf
import io
import sys
import requests
import math
import time

# --- 核心修复：强制控制台使用 UTF-8 ---
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["HF_HOME"] = r"L:\Models\huggingface"  # 设置HF缓存路径

if sys.stdout.encoding != 'utf-8':
    try:
        # Python 3.7+ 的标准修复方法
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # 兼容旧版本
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# --- 配置区 ---
SOURCE_DIRS = [
                # r"D:\2024-网络资源\中文音效",
                # r"E:\sounds\1Archetype低沉撞击",
                # r"E:\sounds\3Endurance低音紧张",
                # r"E:\sounds\【34】变形金刚科幻音效合辑",
                # r"E:\sounds\4Idyllic优雅舒缓",
                # r"E:\sounds\6Beginnings深沉叙事配乐",
                # r"E:\sounds\AI\5000+常用音效",
                # r"E:\sounds\Boom&Rain\Boom_Library_Thunder_Rain_SURR",
                # r"E:\sounds\单独下载",
                # r"D:\2024-网络资源\中文音效",
                #r"E:\sounds\音效素材1",
                #r"E:\sounds\音效素材2",
                #r"E:\sounds\音效素材3",
                #r"E:\sounds\【86】40G Audiojungle资源库",
                #r"E:\sounds\【39】Bluezone公司出品的各类音效",
                #r"E:\sounds\【39】Bluezone公司出品的各类音效",
                #r"E:\sounds\呼吸",
                #r"E:\sounds\呼吸",
                #r"E:\sounds\单独下载",
                # r"L:\BBC-SFX\BBCSoundEffectsComplete\6-30",
                # r"L:\BBC-SFX\BBCSoundEffectsComplete\A-C",
                r"L:\BBC-SFX\BBCSoundEffectsComplete\sounds",
                # r"E:\sounds\战争&科技\Bigfilms CHAOS - Sound FX",
              ]
    # 获取音频文件
audio_exts = ('.wav', '.mp3', '.flac', '.ogg', '.aiff', '.m4a')
#audio_exts = ('.mp3')
TARGET_DIR = r"L:\AI\AI_SFX_BBC_Output"
MODELS_ROOT = r"L:\Models"
HF_CACHE_DIR = os.path.join(MODELS_ROOT, "huggingface")
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
JSON_DB_PATH = os.path.join(TARGET_DIR, "audio_library_v2.json")
OLLAMA_API = "http://localhost:11434/api/generate"
QWEN_MODEL = "qwen:14b-chat-q4_0"

# 【保持当前音效分类体系】
CATEGORY_LIST = [
    "自然环境",           # 所有自然声音统一归类
    "城市环境",           # 所有人造环境声音
    "机械设备",           # 引擎/电机/工业装置
    "生活家居",           # 居家日常音效（榨汁机/马桶/刷牙等）
    "人声",               # 所有语音/非语言人声
    "动物声音",           # 野生动物/昆虫/鸟类
    "冷兵器",             # 刀剑/暗器/格斗音效
    "热兵器",             # 枪械/爆炸/现代武器
    "UI交互",             # 系统/界面/反馈音
    "抽象音效",           # 非现实/电子/变形声
    "转场音效",           # 场景切换/过渡音效
    "电影氛围",           # 心理/情绪铺垫
    "特殊效果",           # 时空/空间/超自然
    "未分类素材"          # 待人工审核
]

# 【专业关键词库】- 用于文件名分析（完整影视标准版）
PROFESSIONAL_KEYWORDS = {
    # 【自然环境】
    "自然环境": [
        # 天气
        "rain", "storm", "wind", "thunder", "lightning", "snow", "hail", "fog", "mist", "drizzle", 
        "blizzard", "hurricane", "typhoon", "tornado", "gale",
        "雨", "风", "雷", "电", "雪", "雹", "雾", "霜", "毛毛雨", "暴雨", 
        "暴风雪", "台风", "龙卷风", "飓风", "狂风",
        
        # 水体
        "ocean", "wave", "sea", "tide", "current", "river", "stream", "brook", "creek", "waterfall",
        "rapids", "drip", "splash", "pour", "flood", "ice", "glacier", "underwater",
        "海", "浪", "潮", "流", "河", "溪", "小溪", "瀑布", "急流", 
        "滴水", "水花", "倾倒", "洪水", "冰", "冰川", "水下",
        
        # 地形与生物
        "forest", "jungle", "desert", "mountain", "cave", "valley", "plateau", "meadow", "grassland",
        "bird", "animal", "insect", "cricket", "frog", "wolf", "lion", "elephant", "bear",
        "森林", "丛林", "沙漠", "山", "洞穴", "山谷", "高原", "草地", "草原",
        "鸟", "动物", "昆虫", "蟋蟀", "青蛙", "狼", "狮子", "大象", "熊"
    ],
    
    # 【城市环境】
    "城市环境": [
        # 交通工具
        "traffic", "car", "bus", "train", "subway", "metro", "tram", "taxi", "truck", "motorcycle",
        "ambulance", "police_car", "fire_truck", "helicopter", "airplane", "jet", "siren", "horn",
        "交通", "汽车", "巴士", "火车", "地铁", "电车", "出租车", "卡车", "摩托车",
        "救护车", "警车", "消防车", "直升机", "飞机", "喷气机", "警笛", "喇叭",
        
        # 人群与建筑
        "crowd", "street", "urban", "city", "downtown", "pedestrian", "market", "shopping", "construction",
        "building", "skyscraper", "bridge", "tunnel", "elevator", "escalator", "staircase", "hallway",
        "人群", "街道", "城市", "都市", "市中心", "行人", "市场", "购物", "施工",
        "建筑", "摩天大楼", "桥", "隧道", "电梯", "扶梯", "楼梯", "走廊"
    ],
    
    # 【机械设备】
    "机械设备": [
        # 动力系统
        "engine", "motor", "turbine", "generator", "compressor", "pump", "fan", "ventilator", "propeller",
        "diesel", "gasoline", "electric", "hydraulic", "pneumatic", "steam", "boiler", "furnace",
        "引擎", "电机", "涡轮", "发电机", "压缩机", "泵", "风扇", "通风机", "螺旋桨",
        "柴油", "汽油", "电力", "液压", "气动", "蒸汽", "锅炉", "熔炉",
        
        # 运动部件
        "gear", "bearing", "chain", "belt", "piston", "crank", "lever", "valve", "sprocket", "rotor",
        "vibration", "grinding", "drilling", "sawing", "cutting", "hammering", "stamping", "pressing",
        "齿轮", "轴承", "链条", "皮带", "活塞", "曲轴", "杠杆", "阀门", "链轮", "转子",
        "振动", "研磨", "钻孔", "锯切", "切割", "锤击", "冲压", "压制"
    ],
    
    # 【生活家居】（新增完整版）
    "生活家居": [
        # 厨房电器
        "juicer", "blender", "mixer", "food_processor", "kitchen_appliance", "refrigerator", "freezer",
        "microwave", "oven", "stove", "cooker", "dishwasher", "sink", "faucet", "tap", "kettle",
        "榨汁机", "搅拌机", "料理机", "厨房电器", "冰箱", "冷冻", "微波炉", "烤箱", "炉灶", 
        "炊具", "洗碗机", "水槽", "水龙头", "水阀", "水壶",
        
        # 卫浴设备
        "toilet", "flush", "bathroom", "shower", "bathtub", "drain", "plumbing", "running_water",
        "toothbrush", "brushing", "shaver", "razor", "hair_dryer", "comb", "mirror", "sink",
        "马桶", "冲水", "卫生间", "淋浴", "浴缸", "排水", "管道", "流水",
        "牙刷", "刷牙", "剃须刀", "刮胡刀", "吹风机", "梳子", "镜子", "洗手池",
        
        # 家居环境
        "door", "door_creak", "window", "lock", "key", "clock", "alarm_clock", "phone", "telephone",
        "doorbell", "vacuum_cleaner", "washing_machine", "dryer", "bed", "bed_spring", "snoring",
        "coughing", "sneezing", "footsteps", "floorboard", "creak", "furniture", "chair", "table",
        "门", "门吱呀", "窗", "锁", "钥匙", "时钟", "闹钟", "电话", 
        "门铃", "吸尘器", "洗衣机", "烘干机", "床", "床弹簧", "打鼾",
        "咳嗽", "打喷嚏", "脚步声", "地板", "吱呀声", "家具", "椅子", "桌子"
    ],
    
    # 【热兵器】（战争/枪械/爆炸）
    "热兵器": [
        # 枪械
        "gun", "rifle", "pistol", "shotgun", "machine_gun", "sniper", "bullet", "ammo", "cartridge",
        "firearm", "trigger", "recoil", "reload", "cock", "safety", "silencer", "suppressor",
        "枪", "步枪", "手枪", "霰弹枪", "机枪", "狙击枪", "子弹", "弹药", "弹匣",
        "火器", "扳机", "后坐力", "装弹", "上膛", "保险", "消音器", "抑制器",
        
        # 爆炸
        "explosion", "bomb", "grenade", "mine", "rocket", "missile", "detonate", "blast", "shockwave",
        "fireball", "debris", "shrapnel", "mushroom_cloud", "booming", "thundering", "rumbling",
        "爆炸", "炸弹", "手榴弹", "地雷", "火箭", "导弹", "引爆", "冲击波",
        "火球", "碎片", "弹片", "蘑菇云", "轰鸣", "雷声", "隆隆声",
        
        # 战争场景
        "war", "battle", "combat", "military", "soldier", "tank", "armored_vehicle", "helicopter_gunship",
        "dogfight", "artillery", "mortar", "howitzer", "cannon", "machine_gun_fire", "rifle_fire",
        "战争", "战役", "战斗", "军事", "士兵", "坦克", "装甲车", "武装直升机",
        "空战", "大炮", "迫击炮", "榴弹炮", "加农炮", "机枪扫射", "步枪射击"
    ],
    
    # 【人声】
    "人声": [
        # 语言
        "speech", "voice", "talk", "dialog", "conversation", "narration", "commentary", "announcement",
        "whisper", "shout", "yell", "scream", "laugh", "cry", "sob", "giggle", "chuckle",
        "对话", "语音", "交谈", "对话", "谈话", "叙述", "解说", "公告",
        "耳语", "喊叫", "叫喊", "尖叫", "笑", "哭", "抽泣", "咯咯笑", "轻笑",
        
        # 非语言
        "breathing", "heavy_breathing", "panting", "gasping", "sigh", "yawn", "cough", "sneeze",
        "footsteps", "footstep", "walking", "running", "jumping", "landing", "clapping", "applause",
        "clothing", "rustle", "zipper", "button", "creak", "squeak", "thud", "thump",
        "呼吸", "急促呼吸", "喘息", "倒吸气", "叹气", "打哈欠", "咳嗽", "打喷嚏",
        "脚步声", "步行", "跑步", "跳跃", "落地", "拍手", "掌声",
        "衣服", "摩擦声", "拉链", "纽扣", "吱呀声", "尖叫声", "重击声", "砰砰声"
    ],
    
    # 【冷兵器】
    "冷兵器": [
        # 兵器类型
        "sword", "blade", "knife", "dagger", "katana", "saber", "cutlass", "machete", "axe", "hammer",
        "mace", "flail", "spear", "lance", "bow", "arrow", "crossbow", "staff", "nunchaku", "chain",
        "刀", "剑", "刀剑", "匕首", "武士刀", "军刀", "水手刀", "砍刀", "斧", "锤",
        "狼牙棒", "连枷", "矛", "长矛", "弓", "箭", "弩", "棍", "双节棍", "链",
        
        # 动作与效果
        "clash", "impact", "strike", "hit", "slash", "stab", "thrust", "parry", "block", "deflect",
        "whoosh", "swish", "slice", "cut", "chop", "smash", "crunch", "shatter", "break",
        "碰撞", "冲击", "打击", "击中", "挥砍", "刺击", "突刺", "格挡", "阻挡", "偏转",
        "嗖声", "嘶嘶声", "切开", "切割", "劈砍", "粉碎", "压碎", "破碎", "断裂"
    ],
    
    # 【UI交互】
    "UI交互": [
        # 基础交互
        "click", "button", "press", "select", "scroll", "drag", "drop", "hover", "menu", "navigation",
        "tap", "touch", "swipe", "pinch", "zoom", "rotate", "gesture", "flick", "slide",
        "点击", "按钮", "按下", "选择", "滚动", "拖动", "放下", "悬停", "菜单", "导航",
        "轻触", "触摸", "滑动", "捏合", "缩放", "旋转", "手势", "轻弹", "滑动",
        
        # 系统反馈
        "beep", "notification", "alert", "error", "warning", "success", "confirm", "cancel", "ding",
        "chime", "ping", "pop", "whoop", "system", "interface", "digital", "electronic", "sonar",
        "哔哔声", "通知", "警报", "错误", "警告", "成功", "确认", "取消", "叮声",
        "钟声", "叮当声", "弹出声", "欢呼声", "系统", "界面", "数字", "电子", "声纳"
    ],
    
    # 【转场音效】
    "转场音效": [
        # 上升/下降
        "riser", "build", "tension_build", "upward", "climax", "ascend", "rise", "sweep_up", "swell",
        "downer", "fall", "release", "drop", "decay", "resolve", "descend", "fade_out", "collapse",
        "上升", "构建", "紧张构建", "向上", "高潮", "上升", "升起", "向上扫频", "膨胀",
        "下降", "坠落", "释放", "掉落", "衰减", "解决", "下降", "淡出", "崩溃",
        
        # 转场效果
        "transition", "whoosh", "sweep", "fly_by", "pass", "slide", "zip", "swish", "crossfade",
        "stinger", "stab", "hit_short", "punctuation", "snap", "pop", "click", "switch", "change",
        "转场", "嗖声", "扫频", "飞过", "经过", "滑动", "拉链", "嘶嘶声", "交叉淡入",
        "强调音", "刺耳声", "短促命中", "标点", "弹响", "爆破声", "点击", "切换", "改变"
    ],
    
    # 【电影氛围】
    "电影氛围": [
        # 情绪铺垫
        "atmosphere", "ambience", "mood", "tension", "suspense", "drama", "romance", "horror", "fear",
        "sadness", "joy", "peace", "calm", "serenity", "loneliness", "isolation", "despair", "hope",
        "氛围", "环境声", "情绪", "紧张", "悬念", "戏剧", "浪漫", "恐怖", "恐惧",
        "悲伤", "喜悦", "和平", "平静", "宁静", "孤独", "孤立", "绝望", "希望",
        
        # 空间氛围
        "spatial", "surround", "dolby", "atmos", "5.1", "7.1", "reverb", "echo", "delay", "depth",
        "distance", "proximity", "perspective", "panning", "movement", "direction", "position", "location",
        "空间", "环绕", "杜比", "全景声", "5.1声道", "7.1声道", "混响", "回声", "延迟", "深度",
        "距离", "接近", "透视", "声像", "移动", "方向", "位置", "定位"
    ],
    
    # 【专业音效类型】（增强版）
    "专业类型": [
        # 基础声学
        "impact", "hit", "strike", "crash", "bang", "thud", "smash", "crunch", "clash", "collide",
        "riser", "build", "upward", "climax", "sweep", "whoosh", "fly_by", "pass", "transition",
        "pulse", "beat", "throb", "heartbeat", "rhythm", "cycle", "stinger", "stab", "accent",
        "drone", "bed", "pad", "atmosphere", "sustained", "background", "texture", "grit", "noise",
        
        # 高级效果
        "glitch", "error", "digital_error", "stutter", "bit_crush", "reverse", "rewind", "backwards",
        "inverse", "flip", "morph", "transform", "shift", "evolve", "change", "granular", "particle",
        "cloud", "diffuse", "sci_fi", "alien", "laser", "energy", "futuristic", "synthetic",
        
        # 中文专业术语
        "冲击", "命中", "打击", "碰撞", "砰", "重击", "粉碎", "压碎", "撞击", "相撞",
        "上升", "构建", "向上", "高潮", "扫频", "嗖声", "飞过", "经过", "转场",
        "脉冲", "节拍", "心跳", "心跳声", "节奏", "循环", "强调音", "刺耳声", "重音",
        "氛围", "铺底", "垫底", "环境氛围", "持续", "背景", "纹理", "颗粒感", "噪音",
        "故障", "错误", "数字错误", "卡顿", "比特压缩", "反转", "倒带", "向后",
        "反向", "翻转", "变形", "转换", "转变", "进化", "变化", "颗粒", "粒子",
        "云", "扩散", "科幻", "外星", "激光", "能量", "未来", "合成"
    ],
    
    # 【动物声音】
    "动物声音": [
        # 野生动物
        "dog", "bark", "howl", "growl", "whine", "cat", "meow", "purr", "hiss", "bird", "chirp",
        "songbird", "eagle", "hawk", "owl", "crow", "raven", "wolf", "howl", "lion", "roar",
        "tiger", "growl", "elephant", "trumpet", "monkey", "chimpanzee", "gorilla", "bear", "roar",
        "dog", "吠叫", "嚎叫", "咆哮", "呜咽", "猫", "喵", "呼噜", "嘶嘶", "鸟", "啁啾",
        "鸣禽", "鹰", "隼", "猫头鹰", "乌鸦", "渡鸦", "狼", "嚎叫", "狮子", "吼叫",
        "老虎", "低吼", "大象", "喇叭声", "猴子", "黑猩猩", "大猩猩", "熊", "吼叫",
        
        # 昆虫与小生物
        "insect", "cricket", "chirp", "cicada", "bee", "buzz", "fly", "mosquito", "frog", "croak",
        "toad", "snake", "hiss", "rat", "squeak", "mouse", "bat", "echolocation", "whale", "song",
        "dolphin", "click", "seagull", "squawk", "chicken", "cluck", "rooster", "crow",
        "昆虫", "蟋蟀", "啁啾", "蝉", "蜜蜂", "嗡嗡声", "苍蝇", "蚊子", "青蛙", "呱呱叫",
        "蟾蜍", "蛇", "嘶嘶声", "老鼠", "吱吱声", "蝙蝠", "声纳定位", "鲸鱼", "歌声",
        "海豚", "咔嗒声", "海鸥", "尖叫声", "鸡", "咯咯声", "公鸡", "打鸣"
    ]
}

# 【关键词权重表】- 不同来源的权重分配
KEYWORD_WEIGHTS = {
    "filename_explicit": 0.9,    # 文件名中明确的专业术语
    "filename_context": 0.7,     # 文件名中的上下文关键词
    "ai_high_confidence": 0.8,   # AI高置信度分析结果
    "ai_medium_confidence": 0.6, # AI中等置信度分析结果
    "ai_low_confidence": 0.4     # AI低置信度分析结果
}


# --- 核心函数：中文验证与清洗 ---
def is_valid_chinese(text):
    """严格验证是否为有效中文（只允许中文、数字、基本标点）"""
    if not text or not isinstance(text, str):
        return False
    
    # 允许的字符范围：
    # \u4e00-\u9fff：中文字符
    # \u3000-\u303f：中文标点
    # \uff00-\uffef：全角字符
    # 0-9：数字
    # 空格
    valid_chars = re.compile(r'^[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef0-9\s、，。；：\'"【】《》（）()\-_]*$')
    
    # 检查是否包含任何英文字符
    if re.search(r'[a-zA-Z]', text):
        return False
    
    # 检查整体格式
    return bool(valid_chars.match(text.strip())) and len(text.strip()) > 0

def clean_chinese_text(text):
    """彻底清洗文本，只保留有效中文字符"""
    if not text:
        return ""
    
    # 1. 移除所有英文字符
    text = re.sub(r'[a-zA-Z]', '', text)
    
    # 2. 移除特殊符号（保留中文标点）
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef0-9\s、，。；：\'"【】《》（）()\-_]', '', text)
    
    # 3. 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. 移除连续重复字符
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    return text

def collect_audio_files(source_dirs, audio_exts):
    """
    从多个源目录中递归收集音频文件。

    :param source_dirs: 源目录列表（str 列表）
    :param audio_exts: 音频扩展名元组，如 ('.wav', '.mp3', '.flac')
    :return: 去重后的绝对路径列表
    """
    if not isinstance(source_dirs, (list, tuple)):
        raise TypeError("source_dirs 必须是列表或元组")

    valid_dirs = []
    for d in source_dirs:
        if not isinstance(d, str):
            print(f"⚠️ 警告：跳过非字符串路径: {d}")
            continue
        if not os.path.isdir(d):
            print(f"⚠️ 警告：路径不存在或不是目录，已跳过: {d}")
            continue
        valid_dirs.append(os.path.abspath(d))  # 转为绝对路径，便于后续处理

    if not valid_dirs:
        print("❌ 错误：没有有效的源目录。")
        return []

    files_set = set()
    for source_dir in valid_dirs:
        for root, _, filenames in os.walk(source_dir):
            for f in filenames:
                if f.lower().endswith(audio_exts):
                    full_path = os.path.abspath(os.path.join(root, f))
                    files_set.add(full_path)

    return sorted(files_set)  # 排序便于调试和日志一致性

def sanitize_filename(filename):
    """安全文件名清洗（保留中文）"""
    filename = re.sub(r'[\\/*?:"<>|]', '', filename).strip()
    filename = filename.replace('_', ' ')
    return filename

def get_file_md5(file_path):
    """计算文件MD5（安全处理大文件）"""
    import hashlib
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"MD5计算失败 {file_path}: {str(e)}")
        return f"error_{abs(hash(str(e)))}"

def preprocess_audio(file_path):
    """预处理音频文件（增强短音效支持）"""
    try:
        import numpy as np
        import librosa
        import math
        from scipy import signal
        
        # 1. 读取音频（兼容各种格式）
        waveform, sample_rate = sf.read(file_path, dtype='float32')
        original_duration = len(waveform) / sample_rate
        print(f"  📏 原始音频时长: {original_duration:.2f}秒")
        
        # 2. 确保是单声道
        if len(waveform.shape) > 1:
            print(f"  🎧 检测到立体声，转换为单声道")
            waveform = waveform.mean(axis=1)
        
        # 3. 处理极短音频（<0.3秒）- 智能循环填充
        MIN_AUDIO_DURATION = 0.3  # 秒
        if original_duration < MIN_AUDIO_DURATION:
            print(f"  ⚠️ 音频过短 ({original_duration:.2f}s)，应用智能循环填充")
            
            # 计算目标样本数
            target_samples = int(MIN_AUDIO_DURATION * sample_rate)
            
            # 智能循环：避免突兀的接缝
            if len(waveform) > 0:
                # 方法1：淡入淡出循环
                fade_samples = min(int(0.02 * sample_rate), len(waveform) // 4)
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                
                # 应用淡入淡出
                if len(waveform) > fade_samples * 2:
                    waveform[:fade_samples] *= fade_in
                    waveform[-fade_samples:] *= fade_out
                
                # 循环填充
                repeats = math.ceil(target_samples / len(waveform))
                extended = np.tile(waveform, repeats)[:target_samples]
                
                # 交叉淡化接缝
                for i in range(1, repeats):
                    start_idx = i * len(waveform) - fade_samples // 2
                    end_idx = i * len(waveform) + fade_samples // 2
                    if start_idx < 0 or end_idx >= len(extended):
                        continue
                    
                    # 交叉淡化
                    cross_fade = np.linspace(1, 0, fade_samples)
                    extended[start_idx:start_idx+fade_samples] *= cross_fade
                    extended[end_idx-fade_samples:end_idx] *= (1 - cross_fade)
                
                waveform = extended
            else:
                # 空音频处理
                waveform = np.zeros(target_samples)
        
        # 4. 重采样到16kHz（AST模型要求）
        if sample_rate != 16000:
            print(f"  🔊 重采样: {sample_rate}Hz → 16000Hz")
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # 5. 智能增强（针对短音效）
        SHORT_AUDIO_THRESHOLD = 1.5  # 秒
        current_duration = len(waveform) / sample_rate
        
        if current_duration < SHORT_AUDIO_THRESHOLD:
            print(f"  🔧 检测到短音效 ({current_duration:.2f}s)，应用专业增强")
            
            # 5.1. 智能时间拉伸（0.3-1.5秒）
            if 0.3 <= current_duration < SHORT_AUDIO_THRESHOLD:
                target_duration = min(SHORT_AUDIO_THRESHOLD, current_duration * 1.8)
                target_samples = int(target_duration * sample_rate)
                
                print(f"  ⏱️ 时间拉伸: {current_duration:.2f}s → {target_duration:.2f}s")
                
                # 使用相位声码器进行高质量时间拉伸
                try:
                    # 方法：使用librosa的time_stretch（高质量）
                    stretch_ratio = current_duration / target_duration
                    waveform = librosa.effects.time_stretch(waveform, rate=stretch_ratio)
                    waveform = waveform[:target_samples]  # 确保长度正确
                    
                    # 如果拉伸后太短，补零
                    if len(waveform) < target_samples:
                        waveform = np.pad(waveform, (0, target_samples - len(waveform)), 'constant')
                    
                    current_duration = len(waveform) / sample_rate
                    print(f"  ✅ 高质量时间拉伸完成，新时长: {current_duration:.2f}s")
                    
                except Exception as e:
                    print(f"  ⚠️ 高质量拉伸失败，使用备用方法: {str(e)}")
                    # 备用方法：线性插值
                    waveform = np.interp(
                        np.linspace(0, len(waveform), target_samples),
                        np.arange(len(waveform)),
                        waveform
                    )
            
            # 5.2. 频谱增强（所有短音效）
            print("  📊 应用频谱增强...")
            try:
                # 高通滤波（移除<80Hz的低频噪声）
                nyquist = sample_rate / 2
                b, a = signal.butter(2, 80 / nyquist, btype='high')
                waveform = signal.filtfilt(b, a, waveform)
                
                # 均衡器增强（提升关键频段）
                freq_ranges = [
                    (200, 500, 1.2),    # 20% 增益 - 基础振动
                    (1000, 4000, 1.3),  # 30% 增益 - 材质细节
                    (5000, 8000, 1.15)  # 15% 增益 - 空间感
                ]
                
                # 使用STFT进行频段增强
                n_fft = 2048
                hop_length = 512
                stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
                freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
                
                for low, high, gain in freq_ranges:
                    # 找到目标频段索引
                    idx = np.where((freqs >= low) & (freqs <= high))[0]
                    if len(idx) > 0:
                        stft[idx, :] *= gain
                
                # 逆STFT
                waveform = librosa.istft(stft, hop_length=hop_length, length=len(waveform))
                
                print("  ✅ 频谱增强完成")
            except Exception as e:
                print(f"  ⚠️ 频谱增强失败: {str(e)}")
            
            # 5.3. 动态范围压缩（提升弱信号）
            try:
                rms = np.sqrt(np.mean(waveform**2))
                if rms > 0.001:  # 避免除零
                    # 压缩比 2:1，阈值 -20dB
                    threshold = 0.1  # -20dB
                    ratio = 2.0
                    
                    mask = np.abs(waveform) > threshold
                    waveform[mask] = threshold + (np.abs(waveform[mask]) - threshold) / ratio * np.sign(waveform[mask])
                    
                    # 提升整体增益
                    waveform *= 1.2
                    
                    print("  📈 动态范围优化完成")
            except Exception as e:
                print(f"  ⚠️ 动态范围处理失败: {str(e)}")
        
        # 6. 归一化到[-1, 1]
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / (max_val + 1e-8)
            print(f"  ✅ 归一化完成，峰值: {max_val:.4f}")
        
        # 7. 确保最小长度（AST模型要求）
        MIN_SAMPLES = 1600  # 0.1秒 at 16kHz
        if len(waveform) < MIN_SAMPLES:
            print(f"  ⚠️ 音频仍过短 ({len(waveform)} samples)，填充到 {MIN_SAMPLES} samples")
            waveform = np.pad(waveform, (0, MIN_SAMPLES - len(waveform)), 'constant')
        
        # 8. 转换为PyTorch张量
        waveform_tensor = torch.tensor(waveform).unsqueeze(0)
        final_duration = waveform_tensor.shape[1] / sample_rate
        print(f"  🎯 预处理完成，最终时长: {final_duration:.2f}秒，形状: {waveform_tensor.shape}")
        
        return waveform_tensor
        
    except ImportError as e:
        print(f"⚠️ 缺少依赖库: {str(e)}")
        print("💡 请安装: pip install numpy librosa scipy")
        # 返回3秒空白音频（兼容模式）
        return torch.zeros(1, 16000 * 3)
    
    except Exception as e:
        print(f"❌ 音频预处理失败 {file_path}: {str(e)}")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误详情: {str(e)}")
        
        # 智能错误恢复
        if "soundfile" in str(e).lower() or "format" in str(e).lower():
            print("  🔄 尝试备用音频加载方法...")
            try:
                import wave
                with wave.open(file_path, 'rb') as wav_file:
                    n_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frame_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    
                    frames = wav_file.readframes(n_frames)
                    if sample_width == 2:  # 16-bit
                        waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    else:  # 8-bit or 24-bit
                        waveform = np.frombuffer(frames, dtype=np.uint8 if sample_width == 1 else np.int32).astype(np.float32)
                        waveform = waveform / (256 if sample_width == 1 else 2147483648.0)
                    
                    if n_channels > 1:
                        waveform = waveform.reshape(-1, n_channels).mean(axis=1)
                    
                    # 重采样到16kHz
                    if frame_rate != 16000:
                        import librosa
                        waveform = librosa.resample(waveform, orig_sr=frame_rate, target_sr=16000)
                    
                    return torch.tensor(waveform).unsqueeze(0)
            except Exception as fallback_e:
                print(f"  ❌ 备用方法也失败: {str(fallback_e)}")
        
        # 最终回退：3秒空白音频
        print("  🛡️ 使用安全回退：3秒空白音频")
        return torch.zeros(1, 16000 * 3)

def generate_readable_filename(description, file_ext, file_hash, max_length=40):
    """生成美观文件名"""
    clean_desc = re.sub(r'[\\/*?:"<>|]', '', description).strip()
    
    # 从描述中提取核心场景（前10个中文字）
    core_scene = ''.join(re.findall(r'[\u4e00-\u9fa5]', clean_desc)[:10])
    
    # 如果提取的场景太短，使用前5个词
    if len(core_scene) < 4:
        words = re.findall(r'[\u4e00-\u9fa5]{1,4}', clean_desc)
        core_scene = ''.join(words[:3]) if words else "音效素材"
    
    short_hash = file_hash[:6]
    filename = f"{core_scene}-{short_hash}{file_ext}"
    
    if len(filename) > max_length:
        filename = filename[:max_length - len(file_ext) - 1] + file_ext
    
    if len(core_scene) < 2:
        filename = f"音效素材-{short_hash}{file_ext}"
    
    return filename

def extract_filename_keywords(original_filename):
    """
    从原始文件名中提取专业关键词
    示例: "01城市灯光交通和步行城市隆隆 01 City,Light Traffic And Pedestrians,City Rumble 天途影像.wav"
    输出: {
        "explicit_keywords": ["交通", "城市隆隆", "traffic", "city rumble"],
        "context_keywords": ["灯光", "步行", "light", "pedestrians"],
        "source_info": ["天途影像"]
    }
    """
    # 1. 移除扩展名和序号
    filename = os.path.splitext(original_filename)[0]
    filename = re.sub(r'^\d+[_\-]?', '', filename).strip()
    
    # 2. 移除来源信息（如"天途影像"）
    source_info = []
    source_match = re.search(r'[_\-\s]([^\s_\-]+影像|sound[_\s]?library|audio[_\s]?archive|recording)[_\-\s]?$', filename, re.IGNORECASE)
    if source_match:
        source_info = [source_match.group(1).strip()]
        filename = filename[:source_match.start()].strip()
    
    # 3. 分割中英文部分
    chinese_parts = re.findall(r'[\u4e00-\u9fa5][^a-zA-Z]*', filename)
    english_parts = re.findall(r'[a-zA-Z][^\\u4e00-\\u9fa5]*', filename)
    
    # 4. 提取显式关键词（通常包含音效类型）
    explicit_keywords = []
    context_keywords = []
    
    # 中文显式关键词
    for part in chinese_parts:
        # 专业音效类型
        for keyword in PROFESSIONAL_KEYWORDS["专业类型"]:
            if keyword in part and len(keyword) >= 2:
                explicit_keywords.append(keyword)
                part = part.replace(keyword, '')
        
        # 其他专业关键词
        words = [w for w in re.split(r'[^\u4e00-\u9fa5]', part) if len(w) >= 2]
        if words:
            # 长度大于3的词更可能是专业术语
            explicit_keywords.extend([w for w in words if len(w) >= 3])
            context_keywords.extend([w for w in words if len(w) == 2])
    
    # 英文显式关键词
    for part in english_parts:
        part = part.lower().strip()
        # 专业音效类型
        for keyword in PROFESSIONAL_KEYWORDS["专业类型"]:
            if keyword.lower() in part:
                clean_keyword = re.sub(r'[^a-z\s]', '', keyword.lower()).strip()
                if clean_keyword and len(clean_keyword) >= 3:
                    explicit_keywords.append(clean_keyword)
        
        # 其他专业关键词
        words = [w.strip() for w in re.split(r'[^a-z]', part) if len(w) >= 2]
        if words:
            explicit_keywords.extend([w for w in words if len(w) >= 4])
            context_keywords.extend([w for w in words if 2 <= len(w) < 4])
    
    # 5. 去重和清洗
    explicit_keywords = list(dict.fromkeys([k.strip() for k in explicit_keywords if k.strip()]))
    context_keywords = list(dict.fromkeys([k.strip() for k in context_keywords if k.strip()]))
    source_info = list(dict.fromkeys([s.strip() for s in source_info if s.strip()]))
    
    return {
        "explicit_keywords": explicit_keywords[:5],  # 最多5个显式关键词
        "context_keywords": context_keywords[:5],    # 最多5个上下文关键词
        "source_info": source_info
    }

def calculate_keyword_confidence(keywords, category):
    """
    计算关键词与分类的匹配置信度
    返回: (confidence_score, matched_keywords)
    """
    if not keywords or not category:
        return 0.0, []
    
    matched_keywords = []
    confidence_score = 0.0
    
    # 1. 检查显式关键词
    for keyword in keywords.get("explicit_keywords", []):
        # 检查是否在专业关键词库中
        for cat, kw_list in PROFESSIONAL_KEYWORDS.items():
            if cat == category and any(kw.lower() in keyword.lower() for kw in kw_list):
                confidence_score += KEYWORD_WEIGHTS["filename_explicit"]
                matched_keywords.append(keyword)
                break
    
    # 2. 检查上下文关键词
    for keyword in keywords.get("context_keywords", []):
        for cat, kw_list in PROFESSIONAL_KEYWORDS.items():
            if cat == category and any(kw.lower() in keyword.lower() for kw in kw_list):
                confidence_score += KEYWORD_WEIGHTS["filename_context"] * 0.8  # 略低权重
                matched_keywords.append(keyword)
                break
    
    # 3. 防止过度自信
    max_possible = len(keywords.get("explicit_keywords", [])) * KEYWORD_WEIGHTS["filename_explicit"] + \
                  len(keywords.get("context_keywords", [])) * KEYWORD_WEIGHTS["filename_context"]
    
    if max_possible > 0:
        confidence_score = min(confidence_score / max_possible, 0.95)
    
    return confidence_score, matched_keywords

class AIEngine:
    def __init__(self, model_name, cache_dir, is_offline, ollama_api, qwen_model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 GPU加速状态: {self.device}") 
        self.ollama_api = ollama_api
        self.qwen_model = qwen_model
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化AST模型
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=is_offline
            )
            self.audio_model = AutoModelForAudioClassification.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=is_offline
            ).to(self.device)
            print(f"✅ 成功加载AST模型: {model_name}")
        except Exception as e:
            print(f"⚠️ AST模型加载失败: {str(e)}")
            self.audio_model = None
            self.feature_extractor = None
            
        # 新增：短音效专用配置
        self.SHORT_AUDIO_THRESHOLD = 1.5  # 秒（短于1.5秒视为短音效）
        self.MIN_AUDIO_DURATION = 0.3     # 秒（最短有效音频）
        
        # 短音效专用关键词库
        self.SHORT_SFX_KEYWORDS = {
            "impact_short": ["click", "tap", "snap", "pop", "thud_short", "hit_short", "stinger", "stab", 
                            "点击", "轻触", "弹响", "爆破", "短促", "强调", "瞬间"],
            "material_short": ["rustle", "crinkle", "crunch_short", "paper", "cloth", "plastic", "foil", 
                              "摩擦", "揉搓", "纸", "布料", "塑料", "锡纸", "薄膜"],
            "nature_short": ["drop", "drip", "splash_small", "leaf_rustle", "bird_chirp_short", 
                            "水滴", "滴落", "小水花", "树叶", "鸟叫短促"],
            "ui_short": ["beep_short", "blip", "bloop", "ding_short", "error_short", "success_short", 
                        "短哔声", "滴答", "提示音", "错误短音", "成功短音"]
        }

    def _enhance_short_audio(self, waveform, sample_rate=16000):
        """
        专业增强短音效（<1.5秒）
        1. 智能时间拉伸（保持音高）
        2. 频谱增强
        3. 循环填充（可选）
        """
        duration = waveform.shape[1] / sample_rate
        
        # 1. 处理极短音频（<0.3秒）
        if duration < self.MIN_AUDIO_DURATION:
            print(f"  ⚠️ 音频过短 ({duration:.2f}s)，使用循环填充")
            target_samples = int(self.MIN_AUDIO_DURATION * sample_rate)
            repeats = math.ceil(target_samples / waveform.shape[1])
            waveform = torch.tile(waveform, (1, repeats))[:, :target_samples]
            return waveform
        
        # 2. 智能时间拉伸（0.3-1.5秒）
        if duration < self.SHORT_AUDIO_THRESHOLD:
            target_duration = min(1.5, duration * 1.8)  # 最多拉伸到1.5秒
            target_samples = int(target_duration * sample_rate)
            
            print(f"  🔧 增强短音效: {duration:.2f}s → {target_duration:.2f}s")
            
            # 使用相位声码器进行高质量时间拉伸（保持音高）
            try:
                import numpy as np
                from scipy.signal import resample
                
                # 转换为numpy进行处理
                audio_np = waveform.squeeze().numpy()
                
                # 高质量重采样（保持音高）
                stretched_audio = resample(audio_np, target_samples)
                
                # 转回PyTorch张量
                waveform = torch.tensor(stretched_audio).unsqueeze(0)
            except Exception as e:
                print(f"  ⚠️ 时间拉伸失败，使用简单重采样: {str(e)}")
                # 备用方案：简单重采样
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0), 
                    size=target_samples, 
                    mode='linear',
                    align_corners=False
                ).squeeze(0)
        
        # 3. 频谱增强（所有短音效）
        if duration < self.SHORT_AUDIO_THRESHOLD:
            try:
                # 应用轻微的均衡器增强（重点提升200Hz-8kHz范围）
                waveform = self._apply_spectral_enhancement(waveform, sample_rate)
            except Exception as e:
                print(f"  ⚠️ 频谱增强失败: {str(e)}")
        
        return waveform

    def _apply_spectral_enhancement(self, waveform, sample_rate=16000):
        """频谱增强：提升短音效的特征清晰度"""
        import numpy as np
        from scipy import signal
        
        audio_np = waveform.squeeze().numpy()
        
        # 1. 应用高通滤波器（移除<80Hz的低频噪声）
        b, a = signal.butter(2, 80/(sample_rate/2), btype='high')
        audio_np = signal.filtfilt(b, a, audio_np)
        
        # 2. 应用均衡器增强（提升关键频段）
        # 200-500Hz: 人声/物体基础
        # 1-4kHz: 材质细节
        # 5-8kHz: 空间感/空气感
        freq_ranges = [
            (200, 500, 1.2),    # 20% 增益
            (1000, 4000, 1.3),  # 30% 增益
            (5000, 8000, 1.15)  # 15% 增益
        ]
        
        for low, high, gain in freq_ranges:
            # 使用FFT进行频段增强
            spectrum = np.fft.rfft(audio_np)
            freqs = np.fft.rfftfreq(len(audio_np), 1/sample_rate)
            
            # 找到目标频段索引
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            if len(idx) > 0:
                spectrum[idx] *= gain
        
            # 逆变换回时域
            audio_np = np.fft.irfft(spectrum, n=len(audio_np))
        
        # 3. 动态范围压缩（提升弱信号）
        rms = np.sqrt(np.mean(audio_np**2))
        if rms > 0:
            # 对低于RMS的信号进行提升
            mask = np.abs(audio_np) < rms
            audio_np[mask] *= 1.5  # 50% 增益
        
        # 4. 限制峰值防止失真
        max_val = np.max(np.abs(audio_np))
        if max_val > 0.99:
            audio_np = audio_np * 0.99 / max_val
        
        return torch.tensor(audio_np).unsqueeze(0)

    def _calculate_spectral_centroid(self, waveform, sample_rate=16000):
        """计算频谱质心"""
        try:
            if waveform.shape[1] < 1024:
                return 0
            
            segment = waveform[0, :1024].numpy()
            spectrum = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), 1/sample_rate)
            spectral_centroid = np.sum(spectrum * freqs) / np.sum(spectrum)
            return float(spectral_centroid)
        except:
            return 0

    def _generate_acoustic_fingerprint(self, waveform, sample_rate=16000):
        """生成声学指纹"""
        try:
            duration = waveform.shape[1] / sample_rate
            rms = torch.sqrt(torch.mean(waveform**2)).item()
            spectral_centroid = self._calculate_spectral_centroid(waveform, sample_rate)
            
            fingerprint = (
                f"持续时间:{duration:.1f}s, "
                f"动态范围:{rms*100:.1f}%, "
                f"主导频率:{spectral_centroid:.0f}Hz"
            )
            return fingerprint
        except Exception as e:
            print(f"⚠️ 声学指纹生成失败: {str(e)}")
            return "专业音效特征, 通用频谱分析"

    def classify_audio(self, waveform, categories, original_filename):
        """融合文件名分析的精准分类（增强短音效支持）"""
        if self.audio_model is None:
            return "未分类素材", 0.0
        
        try:
            # 1. 检测音频时长
            duration = waveform.shape[1] / 16000  # sample_rate=16000
            
            # 2. 短音效专项处理
            is_short_audio = duration < self.SHORT_AUDIO_THRESHOLD
            original_waveform = waveform.clone() if is_short_audio else None
            
            if is_short_audio:
                print(f"⏱️ 检测到短音效: {duration:.2f}秒")
                waveform = self._enhance_short_audio(waveform)
            
            # 3. AST模型分析
            inputs = self.feature_extractor(
                waveform.squeeze().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.audio_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                _, top_idxs = probs[0].topk(3)
                confidence_scores = probs[0][top_idxs].cpu().numpy()
            
            ast_labels = [self.audio_model.config.id2label[idx.item()] for idx in top_idxs]
            ast_category = self._map_to_simple_category(ast_labels, categories)
            ast_confidence = float(confidence_scores[0])
            
            # 4. 文件名关键词分析
            filename_keywords = extract_filename_keywords(original_filename)
            filename_category, filename_confidence = self._analyze_filename_for_category(
                filename_keywords, categories
            )
            
            # 5. 融合决策
            final_category, final_confidence = self._fuse_decisions_short_audio(
                ast_category, ast_confidence,
                filename_category, filename_confidence,
                is_short_audio, duration,
                filename_keywords, ast_labels
            )
            
            # 6. 低置信度修复
            final_category, final_confidence = self._repair_low_confidence(
                waveform, final_category, final_confidence, filename_keywords
            )
            
            print(f"🎯 短音效融合: 原始({ast_category}, {ast_confidence:.2f}) + 文件名({filename_category}, {filename_confidence:.2f}) → {final_category}, {final_confidence:.2f}")
            
            return final_category, final_confidence
            
        except Exception as e:
            print(f"分类失败: {str(e)}")
            return "未分类素材", 0.0

    def _fuse_decisions_short_audio(self, ast_category, ast_confidence, 
                                   filename_category, filename_confidence,
                                   is_short_audio, duration,
                                   filename_keywords, ast_labels):
        """
        短音效专用决策融合
        原则：
        1. 短音效优先信任文件名
        2. 低置信度时使用保守策略
        3. 识别特殊短音效类型（impact/材质/自然）
        """
        # 1. 首先检查是否为特殊短音效类型
        short_sfx_category = self._detect_short_sfx_type(filename_keywords, ast_labels, duration)
        if short_sfx_category and short_sfx_category in ["转场音效", "UI交互", "冲击音效_impact", "自然环境"]:
            print(f"  🎯 识别为特殊短音效类型: {short_sfx_category}")
            return short_sfx_category, 0.85
    
        # 2. 短音效决策规则
        if is_short_audio:
            # 规则1: 文件名置信度 > 0.6 时，优先使用文件名分类
            if filename_confidence > 0.6:
                return filename_category, filename_confidence * 0.9
            
            # 规则2: AST置信度 < 0.5 且文件名有明确关键词，使用文件名分类
            if ast_confidence < 0.5 and filename_confidence > 0.3:
                return filename_category, max(ast_confidence, filename_confidence) * 0.85
            
            # 规则3: 两者都不确定，使用保守分类
            if ast_confidence < 0.4 and filename_confidence < 0.4:
                return self._get_conservative_category(filename_keywords, ast_labels), 0.4
    
        # 3. 默认回退到原始融合逻辑
        return self._fuse_decisions(ast_category, ast_confidence, filename_category, filename_confidence)

    def _fuse_decisions(self, ast_category, ast_confidence, filename_category, filename_confidence):
        """融合AST和文件名的决策"""
        # 1. 如果两者一致，取最高置信度
        if ast_category == filename_category:
            return ast_category, max(ast_confidence, filename_confidence)
        
        # 2. 计算加权置信度
        weighted_ast = ast_confidence * 0.7  # AST权重70%
        weighted_filename = filename_confidence * 0.3  # 文件名权重30%
        
        # 3. 特殊规则：文件名包含明确专业术语时，优先考虑文件名
        if filename_confidence > 0.7 and ast_confidence < 0.5:
            return filename_category, filename_confidence
        
        # 4. 默认规则：取加权置信度高的
        if weighted_ast > weighted_filename:
            return ast_category, ast_confidence
        else:
            return filename_category, filename_confidence

    def _detect_short_sfx_type(self, filename_keywords, ast_labels, duration):
        """检测特殊短音效类型"""
        # 提取所有关键词
        all_keywords = filename_keywords.get("explicit_keywords", []) + \
                      filename_keywords.get("context_keywords", []) + \
                      [label.lower() for label in ast_labels]
        
        all_keywords_str = " ".join(all_keywords).lower()
        
        # 1. 检查冲击/强调音效
        impact_keywords = ["click", "tap", "snap", "pop", "thud", "hit", "stinger", "stab", 
                          "点击", "轻触", "弹响", "爆破", "强调", "瞬间"]
        if any(kw in all_keywords_str for kw in impact_keywords) and duration < 1.0:
            return "转场音效"  # 使用现有分类
        
        # 2. 检查UI交互音效
        ui_keywords = ["button", "beep", "notification", "system", "ui", "界面", "系统", "提示"]
        if any(kw in all_keywords_str for kw in ui_keywords) and duration < 0.8:
            return "UI交互"
        
        # 3. 检查材质音效（纸/布/塑料）
        material_keywords = ["paper", "cloth", "plastic", "foil", "rustle", "crinkle", 
                            "纸", "布料", "塑料", "锡纸", "摩擦", "揉搓"]
        if any(kw in all_keywords_str for kw in material_keywords) and duration < 1.2:
            return "自然环境"  # 或者创建"材质交互"类别，但使用现有
        
        # 4. 检查自然短音效
        nature_keywords = ["drop", "drip", "splash", "leaf", "bird_chirp", 
                          "水滴", "滴落", "水花", "树叶", "鸟叫"]
        if any(kw in all_keywords_str for kw in nature_keywords) and duration < 0.7:
            return "自然环境"
        
        return None

    def _get_conservative_category(self, filename_keywords, ast_labels):
        """保守分类策略：低置信度时使用"""
        # 1. 检查文件名中的明确类别指示
        explicit_keys = " ".join(filename_keywords.get("explicit_keywords", [])).lower()
        
        if any(kw in explicit_keys for kw in ["ui", "button", "click", "界面", "按钮", "点击"]):
            return "UI交互"
        if any(kw in explicit_keys for kw in ["impact", "hit", "thud", "冲击", "打击", "砰"]):
            return "转场音效"  # 使用现有分类
        if any(kw in explicit_keys for kw in ["paper", "cloth", "material", "纸", "布", "材质"]):
            return "自然环境"
        if any(kw in explicit_keys for kw in ["juicer", "blender", "toilet", "flush", "toothbrush", "榨汁", "马桶", "刷牙"]):
            return "生活家居"
        
        # 2. 最后回退
        return "未分类素材"

    def _repair_low_confidence(self, waveform, base_category, confidence, filename_keywords):
        """智能修复低置信度分类"""
        if confidence < 0.5:
            print(f"  🔧 修复低置信度 ({confidence:.2f})...")
            
            # 策略1: 优先使用文件名关键词
            filename_category, filename_confidence = self._analyze_filename_for_category(
                filename_keywords, CATEGORY_LIST
            )
            
            if filename_confidence > confidence * 1.5:
                print(f"  ✅ 用文件名覆盖: {base_category}({confidence:.2f}) → {filename_category}({filename_confidence:.2f})")
                return filename_category, filename_confidence
            
            # 策略2: 检查短音效类型
            duration = waveform.shape[1] / 16000  # 计算音频时长
            short_sfx_type = self._detect_short_sfx_type(filename_keywords, [base_category], duration)
            if short_sfx_type:
                print(f"  ✅ 识别为短音效类型: {short_sfx_type}")
                return short_sfx_type, 0.8
            
            # 策略3: 保守回退
            print(f"  ⚠️ 无法修复，使用保守分类: 未分类素材")
            return "未分类素材", 0.4
        
        return base_category, confidence

    def _map_to_simple_category(self, ast_labels, target_categories):
        """精准映射逻辑"""
        mapping_rules = {
            "自然环境": ["wave", "rain", "wind", "thunder", "ocean", "bird", "animal", "forest", "water", "nature",
                        "stream", "river", "brook", "creek", "waterfall", "drip", "splash"],
            "城市环境": ["traffic", "car", "bus", "train", "subway", "siren", "crowd", "street", "urban",
                        "construction", "drill", "jackhammer", "horn", "ambulance", "police_siren"],
            "机械设备": ["engine", "machine", "drill", "saw", "fan", "motor", "tools", "mechanical",
                        "industrial", "factory", "generator", "compressor", "pump", "turbine"],
            "生活家居": ["juicer", "blender", "toilet", "flush", "toothbrush", "bathroom", "kitchen", "appliance",
                        "home", "house", "domestic", "appliances", "refrigerator", "microwave"],
            "人声": ["speech", "voice", "talk", "dialog", "narration", "laugh", "cry", "human",
                    "breathing", "footsteps", "clothing", "rustle", "whisper", "sigh", "yawn"],
            "冷兵器": ["sword", "blade", "knife", "metal", "clash", "impact", "whoosh", "kungfu", "martial",
                    "samurai", "swordsmen", "katana", "dagger", "spear"],
            "热兵器": ["gun", "rifle", "shotgun", "machine_gun", "sniper", "bullet", "explosion", "bomb", "grenade",
                    "firearm", "ammunition", "detonation", "war", "military", "combat"],
            "动物声音": ["dog", "cat", "bird", "animal", "insect", "cricket", "wildlife",
                        "lion", "tiger", "elephant", "monkey", "chimpanzee", "frog"],
            "UI交互": ["click", "button", "beep", "notification", "interface", "digital", "ui",
                    "menu", "scroll", "typing", "keyboard", "mouse", "hover"],
            "转场音效": ["transition", "whoosh", "sweep", "riser", "downer", "swish", "effect",
                        "fade", "crossfade", "stinger", "impact_short"],
            "电影氛围": ["atmosphere", "ambient", "mood", "tension", "calm", "emotional",
                        "suspense", "drama", "romance", "horror", "peaceful"],
            "特殊效果": ["sci-fi", "magic", "fantasy", "special", "unusual",
                        "space", "alien", "laser", "energy", "futuristic"]
        }
        
        for category, keywords in mapping_rules.items():
            if any(keyword in label.lower() for label in ast_labels for keyword in keywords):
                if category in target_categories:
                    return category
        
        label_str = " ".join(ast_labels).lower()
        if "water" in label_str or "wave" in label_str or "ocean" in label_str:
            return "自然环境"
        if "traffic" in label_str or "car" in label_str or "urban" in label_str:
            return "城市环境"
        if "engine" in label_str or "machine" in label_str or "motor" in label_str:
            return "机械设备"
        if "juice" in label_str or "toilet" in label_str or "tooth" in label_str or "home" in label_str:
            return "生活家居"
        
        return "未分类素材"

    def _analyze_filename_for_category(self, filename_keywords, target_categories):
        """从文件名关键词分析分类"""
        max_confidence = 0.0
        best_category = "未分类素材"
        
        # 1. 检查显式关键词
        for category in target_categories:
            confidence, _ = calculate_keyword_confidence(filename_keywords, category)
            if confidence > max_confidence:
                max_confidence = confidence
                best_category = category
        
        # 2. 如果置信度低，使用启发式规则
        if max_confidence < 0.3:
            explicit_keys = " ".join(filename_keywords.get("explicit_keywords", [])).lower()
            context_keys = " ".join(filename_keywords.get("context_keywords", [])).lower()
            
            if any(kw in explicit_keys+context_keys for kw in ["rain", "storm", "wind", "thunder", "雨", "风", "雷"]):
                return "自然环境", 0.6
            if any(kw in explicit_keys+context_keys for kw in ["traffic", "car", "bus", "train", "交通", "汽车", "巴士", "火车"]):
                return "城市环境", 0.6
            if any(kw in explicit_keys+context_keys for kw in ["juicer", "blender", "toilet", "flush", "toothbrush", 
                                                            "榨汁", "搅拌机", "马桶", "冲水", "刷牙"]):
                return "生活家居", 0.7
            if any(kw in explicit_keys+context_keys for kw in ["sword", "blade", "metal", "clash", "刀", "剑", "金属", "碰撞"]):
                return "冷兵器", 0.7
            if any(kw in explicit_keys+context_keys for kw in ["gun", "rifle", "explosion", "gun", "rifle", "爆炸", "枪声"]):
                return "热兵器", 0.7
        
        return best_category, min(max_confidence, 0.85)  # 限制最大置信度

    def get_semantic_tags(self, waveform, base_category, original_filename, classification_confidence):
        """生成增强版语义标签（融合文件名信息）"""
        acoustic_fingerprint = self._generate_acoustic_fingerprint(waveform)
        
        # 1. 从文件名提取关键词
        filename_keywords = extract_filename_keywords(original_filename)
        filename_confidence, matched_keywords = calculate_keyword_confidence(filename_keywords, base_category)
        
        # 2. 构建增强版提示词
        filename_context = ""
        if matched_keywords:
            filename_context = f"文件名关键词: {', '.join(matched_keywords)}"
        
        # 3. 动态调整提示词详细程度
        detail_level = "详细" if classification_confidence > 0.7 else "中等"
        focus_area = "重点描述文件名中提到的元素" if filename_confidence > 0.5 else ""
        
        prompt = f"""
🎯 任务要求
你是一位专业音效设计师，融合声学特征、分类信息和原始文件名关键词，生成：
1. 一段自然语言的场景描述 (30字以内，有画面感，重点信息放在最前面)
2. 1-6个中文标签 (覆盖: 场景分类, 场景内容, 音效类型, 具体事物)
3. 1-3个对应的英文标签 (专业术语)

注意：
1.你对最终音效内容的判断不要过渡依赖文件名，如果文件名描述不清晰，或是无规则文件名，就直接忽略文件名不做参考
2.长度低于1.5秒的文件 TAG附加标注“短音效”
3.如果有能直接翻译成中文的单个英文单词的文件名，就把它加入到TAG里。

📁 基础分类
{base_category} (置信度: {classification_confidence:.2f})

📄 原始文件名信息
原始文件名: {original_filename}
{filename_context}

🔍 任务要求
- 置信度高时: {detail_level}描述，包含专业细节
- {focus_area}
- 避免与文件名关键词明显矛盾的描述
- 所有中文标签必须是纯中文，不包含任何英文字符或数字
- 标签长度控制在2-6个中文字符

✅ 输出格式 (严格遵循)
场景描述: [50字以内的自然语言描述]
中文标签: [词1, 词2, 词3, 词4, 词5, 词6]
英文标签: [term1, term2, term3]

📝 示例1
场景描述: 深夜暴雨倾盆而下，雨点砸在竹叶上发出沙沙声，远处传来隆隆雷声
中文标签: 暴雨, 竹林, 雨滴, 雷声, 夜晚, 自然环境, 水体, 气象, 户外
英文标签: heavy_rain, bamboo_forest, rain_drops

📝 示例2
场景描述: 两匹马在草原上奔跑
中文标签: 自然环境, 马, 动物自然, 马蹄声, 草原
英文标签: Horses, natural, grassland

🎯 当前任务
"""
        
        try:
            payload = {
                "model": self.qwen_model, 
                "prompt": prompt, 
                "stream": False,
                "options": {
                    "temperature": 0.2 if classification_confidence > 0.7 else 0.3,
                    "num_ctx": 2048
                }
            }
            response = requests.post(
                self.ollama_api, 
                json=payload, 
                timeout=40
            )
            response.raise_for_status()
            
            content = response.json()['response'].strip()
            
            # 清理思考过程
            for marker in ["</think>", "任务结束", "思考过程"]:
                if marker in content:
                    content = content.split(marker)[-1].strip()
            
            # 解析结构化输出
            desc_match = re.search(r"场景描述:\s*(.*)", content)
            cn_match = re.search(r"中文标签:\s*(.*)", content)
            en_match = re.search(r"英文标签:\s*(.*)", content)
            
            # 场景描述 (50字以内)
            description = desc_match.group(1).strip() if desc_match else f"{base_category}专业音效素材"
            if len(description) > 50:
                description = description[:50]
            
            # 中文标签 (3-9个)
            if cn_match:
                raw_tags_cn = cn_match.group(1).strip()
                tags_cn = [tag.strip() for tag in raw_tags_cn.split(",") if tag.strip()]
                # 彻底清洗并验证
                tags_cn = self._clean_chinese_tags(tags_cn)
            else:
                tags_cn = ["未知音效"]
            
            # 英文标签 (3-9个，对应中文)
            if en_match:
                raw_tags_en = en_match.group(1).strip()
                tags_en = [tag.strip() for tag in raw_tags_en.split(",") if tag.strip()]
                tags_en = tags_en[:len(tags_cn)] if len(tags_en) > len(tags_cn) else (tags_en + ["professional_sound"] * (len(tags_cn) - len(tags_en)))[:9]
            else:
                tags_en = ["professional_sound", "film_sound", "sound_design"]
            
            # 二次清洗
            tags_cn = [clean_chinese_text(tag) for tag in tags_cn if is_valid_chinese(tag)]
            if len(tags_cn) < 3:
                default_tags = ["未知音效"]
                for tag in default_tags:
                    if tag not in tags_cn:
                        tags_cn.append(tag)
                    if len(tags_cn) >= 3:
                        break
            
            # 质量验证
            if len(description) < 10:
                description = self._get_fallback_description(base_category)
            
            return description, tags_cn, tags_en
            
        except Exception as e:
            print(f"⚠️ Qwen3 API 失败: {str(e)}")
            fallback_desc, fallback_cn, fallback_en = self._get_fallback_tags(base_category)
            fallback_cn = [clean_chinese_text(tag) for tag in fallback_cn if is_valid_chinese(tag)]
            return fallback_desc, fallback_cn, fallback_en

    def _clean_chinese_tags(self, tags):
        """彻底清洗中文标签，确保100%中文"""
        cleaned_tags = []
        
        for tag in tags:
            # 1. 彻底清洗文本
            clean_tag = clean_chinese_text(tag)
            
            # 2. 验证是否为有效中文
            if is_valid_chinese(clean_tag):
                # 3. 限制长度（2-6个中文字符）
                clean_tag = clean_tag[:6]
                if len(clean_tag) >= 2:
                    cleaned_tags.append(clean_tag)
        
        # 4. 确保至少有3个标签
        if len(cleaned_tags) < 3:
            default_tags = ["未知音效"]
            for tag in default_tags:
                if tag not in cleaned_tags:
                    cleaned_tags.append(tag)
                if len(cleaned_tags) >= 3:
                    break
        
        # 5. 限制最多9个标签
        return cleaned_tags[:9]

    def _fuse_filename_keywords(self, tags_cn, tags_en, filename_keywords, category):
        """将文件名关键词智能融合到标签中"""
        # 1. 提取关键文件名关键词
        explicit_keys = filename_keywords.get("explicit_keywords", [])
        context_keys = filename_keywords.get("context_keywords", [])
        
        # 2. 优先级：专业类型关键词 > 具体内容关键词 > 通用关键词
        all_keys = explicit_keys + context_keys
        
        # 3. 筛选与分类相关的关键词
        relevant_keys = []
        for key in all_keys:
            # 检查是否与当前分类相关
            for cat, kw_list in PROFESSIONAL_KEYWORDS.items():
                if cat == category and any(kw.lower() in key.lower() for kw in kw_list):
                    relevant_keys.append(key)
                    break
        
        # 4. 去重和清洗
        relevant_keys = list(dict.fromkeys([k.strip() for k in relevant_keys if k.strip()]))
        
        # 5. 智能融合
        if relevant_keys:
            # 将最相关的2个关键词插入到标签前部
            new_tags_cn = relevant_keys[:2] + [t for t in tags_cn if t not in relevant_keys][:7]
            # 英文标签保持原有逻辑，但确保数量匹配
            new_tags_en = [self._translate_to_english(k) for k in relevant_keys[:2]] + \
                         [t for t in tags_en if t not in [self._translate_to_english(k) for k in relevant_keys]][:7]
            
            # 去重
            new_tags_cn = list(dict.fromkeys(new_tags_cn))[:9]
            new_tags_en = list(dict.fromkeys(new_tags_en))[:9]
            
            return new_tags_cn, new_tags_en
        
        return tags_cn, tags_en

    def _translate_to_english(self, chinese_text):
        """简单中译英（专业音效术语）"""
        translation_map = {
            # 自然
            "雨": "rain", "暴雨": "heavy_rain", "风": "wind", "雷": "thunder", "海浪": "ocean_wave",
            "水": "water", "溪流": "stream", "森林": "forest", "鸟": "bird", "虫": "insect",
            
            # 城市
            "交通": "traffic", "汽车": "car", "人群": "crowd", "街道": "street", "城市": "city",
            
            # 生活家居
            "榨汁": "juicer", "搅拌机": "blender", "马桶": "toilet", "冲水": "flush", "刷牙": "brushing_teeth",
            "牙刷": "toothbrush", "厨房": "kitchen", "卫浴": "bathroom", "家居": "household",
            
            # 冷兵器
            "刀": "sword", "剑": "sword", "刀剑": "sword", "金属": "metal", "碰撞": "clash",
            "打击": "impact", "功夫": "kungfu", "武术": "martial_arts", "武士": "warrior",
            
            # 热兵器
            "枪": "gun", "步枪": "rifle", "手枪": "pistol", "爆炸": "explosion", "炸弹": "bomb",
            "手榴弹": "grenade", "军事": "military", "战争": "war", "战斗": "combat",
            
            # 专业类型
            "冲击": "impact", "上升": "riser", "转场": "transition", "脉冲": "pulse", "氛围": "atmosphere",
            "强调": "stinger", "过渡": "crossfade", "环境": "ambience"
        }
        
        # 精确匹配
        if chinese_text in translation_map:
            return translation_map[chinese_text]
        
        # 部分匹配
        for zh, en in translation_map.items():
            if zh in chinese_text:
                return en
        
        # 默认处理：小写+下划线
        return re.sub(r'[^\w\s]', '', chinese_text).lower().replace(' ', '_')

    def _get_fallback_tags(self, base_category):
        """智能回退方案（确保中文标签纯净）"""
        fallbacks = {
            "自然环境": (
                "深夜森林中，微风吹过树叶发出沙沙声，远处传来潺潺流水和虫鸣",
                ["森林", "微风", "树叶", "流水", "虫鸣", "夜晚", "自然环境", "水体", "生物"],
                ["forest", "breeze", "leaves", "stream", "insects", "night", "natural_environment", "water_body", "wildlife"]
            ),
            "城市环境": (
                "繁忙的都市街道，汽车喇叭声、人群交谈声和远处的施工声交织在一起",
                ["街道", "交通", "人群", "城市", "车辆", "喇叭", "施工", "都市", "环境声"],
                ["street", "traffic", "crowd", "city", "vehicles", "horn", "construction", "urban", "ambience"]
            ),
            "机械设备": (
                "老旧柴油发动机在厂房内持续运转，发出低沉的轰鸣和机械零件的规律碰撞声",
                ["引擎", "机械", "工业", "柴油", "运转", "轰鸣", "厂房", "设备", "动力系统"],
                ["engine", "mechanical", "industrial", "diesel", "operation", "roar", "factory", "equipment", "power_system"]
            ),
            "生活家居": (
                "清晨厨房中，榨汁机嗡嗡运转，新鲜水果被搅拌成汁，旁边水龙头滴着水珠",
                ["榨汁机", "厨房", "早晨", "水果", "家用电器", "生活场景", "日常", "家居", "早餐"],
                ["juicer", "kitchen", "morning", "fruit", "household_appliance", "daily_life", "home", "domestic", "breakfast"]
            ),
            "冷兵器": (
                "竹林中两名武士持刀对决，金属刀身碰撞发出清脆响声，衣袂随风飘动",
                ["刀剑", "碰撞", "金属", "竹林", "武士", "对决", "冷兵器", "传统", "武侠"],
                ["sword", "clash", "metal", "bamboo_forest", "warrior", "duel", "cold_weapon", "traditional", "martial_arts"]
            ),
            "热兵器": (
                "战场上的M16步枪连发射击，子弹呼啸而过，远处爆炸声震耳欲聋",
                ["步枪", "射击", "战场", "爆炸", "军事", "战争", "M16", "子弹", "爆炸声"],
                ["rifle", "shooting", "battlefield", "explosion", "military", "war", "m16", "bullet", "explosion_sound"]
            ),
            "人声": (
                "紧张的密室对话，两人低声交谈，呼吸急促，偶尔有衣物摩擦声",
                ["对话", "密室", "紧张", "呼吸", "低语", "人声", "情绪", "非语言", "氛围"],
                ["dialog", "secret_room", "tension", "breathing", "whisper", "human_voice", "emotion", "non_verbal", "atmosphere"]
            ),
            "动物声音": (
                "清晨热带雨林，各种鸟类鸣叫，昆虫嗡嗡，远处有猴子的叫声",
                ["鸟鸣", "昆虫", "猴子", "雨林", "清晨", "野生动物", "自然", "生物", "生态环境"],
                ["bird_song", "insects", "monkey", "rainforest", "morning", "wildlife", "nature", "creature", "ecosystem"]
            ),
            "UI交互": (
                "未来科技界面，按钮点击时发出清脆的确认声，伴随微妙的触觉反馈和视觉提示音",
                ["按钮", "系统", "确认", "界面", "触觉", "未来", "科技", "反馈", "交互"],
                ["button", "system", "confirmation", "interface", "haptic", "futuristic", "technology", "feedback", "interaction"]
            ),
            "转场音效": (
                "电影场景从安静的室内切换到狂风暴雨的户外，使用上升音效和风声渐变过渡",
                ["转场", "上升", "风声", "渐变", "电影", "场景", "过渡", "动态", "效果"],
                ["transition", "riser", "wind", "fade", "cinema", "scene", "crossfade", "dynamic", "effect"]
            ),
            "电影氛围": (
                "悬疑场景的紧张氛围铺垫，低频持续音效配合细微的心跳声，营造不安情绪",
                ["氛围", "悬疑", "铺垫", "低频", "心跳", "紧张", "情绪", "心理", "电影"],
                ["atmosphere", "suspense", "buildup", "low_frequency", "heartbeat", "tension", "emotion", "psychological", "cinema"]
            ),
            "特殊效果": (
                "科幻空间扭曲效果，能量波动产生嗡嗡声，伴随粒子消散的嘶嘶声",
                ["科幻", "空间", "能量", "扭曲", "粒子", "未来", "超自然", "变形", "特效"],
                ["sci_fi", "space", "energy", "distortion", "particles", "futuristic", "supernatural", "morph", "special_effect"]
            ),
            "未分类素材": (
                "专业音效素材，适用于多种影视和游戏场景，具有独特的声学特征",
                ["专业", "音效", "素材", "影视", "游戏", "通用", "设计", "创意", "资源"],
                ["professional", "sound_effect", "material", "film", "game", "versatile", "design", "creative", "resource"]
            )
        }
        
        # 确保回退标签是纯中文
        if base_category in fallbacks:
            desc, cn_tags, en_tags = fallbacks[base_category]
            return desc, cn_tags, en_tags
        else:
            return (
                f"{base_category}专业音效场景描述",
                ["专业音效", "影视素材", "音效设计", "创意声音", "专业制作"],
                ["professional_sound", "film_material", "sound_design", "creative_audio", "professional_production"]
            )

    def _get_fallback_description(self, base_category):
        """回退描述"""
        descriptions = {
            "自然环境": "深夜森林中，微风吹过树叶发出沙沙声，远处传来潺潺流水和虫鸣",
            "城市环境": "繁忙的都市街道，汽车喇叭声、人群交谈声和远处的施工声交织在一起",
            "机械设备": "老旧柴油发动机在厂房内持续运转，发出低沉的轰鸣和机械零件的规律碰撞声",
            "生活家居": "清晨厨房中，榨汁机嗡嗡运转，新鲜水果被搅拌成汁，旁边水龙头滴着水珠",
            "冷兵器": "竹林中两名武士持刀对决，金属刀身碰撞发出清脆响声，衣袂随风飘动",
            "热兵器": "战场上的M16步枪连发射击，子弹呼啸而过，远处爆炸声震耳欲聋",
            "人声": "紧张的密室对话，两人低声交谈，呼吸急促，偶尔有衣物摩擦声"
        }
        return descriptions.get(base_category, f"{base_category}专业音效场景")

def main():
    start_time = time.time()  # 记录开始时间
    
    # 检查目标目录
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 初始化引擎
    engine = AIEngine(AST_MODEL, HF_CACHE_DIR, True, OLLAMA_API, QWEN_MODEL)
    
    # 加载数据库
    db_data = []
    if os.path.exists(JSON_DB_PATH):
        try:
            with open(JSON_DB_PATH, 'r', encoding='utf-8') as f:
                db_data = json.load(f)
            print(f"✅ 已加载 {len(db_data)} 个已处理文件记录")
        except Exception as e:
            print(f"⚠️ 读取历史记录失败: {str(e)}")
    processed_hashes = {item['md5'] for item in db_data}

    # files = [
    #     os.path.join(root, f)
    #     for root, _, filenames in os.walk(SOURCE_DIR)
    #     for f in filenames if f.lower().endswith(audio_exts)
    # ]

    # 收集文件
    files = collect_audio_files(SOURCE_DIRS, audio_exts)

    print(f"\n{'='*60}")
    print(f"🎯 任务配置:")
    print(f"   源目录: {SOURCE_DIRS}")
    print(f"   目标目录: {TARGET_DIR}")
    print(f"   文件总数: {len(files)}")
    print(f"   已处理: {len(processed_hashes)}")
    print(f"   待处理: {len(files) - len(processed_hashes)}")
    print(f"   智能融合: 原始文件名 + AI分析")
    print(f"{'='*60}")
    
    # 处理文件
    processed_count = 0
    for f_path in tqdm(files, desc="整理中", ascii=True):
        try:
            f_hash = get_file_md5(f_path)
            if f_hash in processed_hashes:
                continue
            
            # 获取原始文件名
            original_filename = os.path.basename(f_path)
            
            waveform = preprocess_audio(f_path)
            
            # 1. 融合分类（包含文件名分析）
            initial_cat, classification_confidence = engine.classify_audio(
                waveform, CATEGORY_LIST, original_filename
            )
            
            # 2. 置信度<0.5时强制重定向到未分类目录
            original_cat_for_stats = initial_cat  # 保留原始分类用于统计
            if classification_confidence < 0.5 and initial_cat != "未分类素材":
                print(f"  ⚠️ 低置信度 ({classification_confidence:.2f})，强制重定向到: 未分类素材")
                initial_cat = "未分类素材"
            
            # 3. 生成增强标签（融合文件名信息）
            description, tags_cn, tags_en = engine.get_semantic_tags(
                waveform, initial_cat, original_filename, classification_confidence
            )
            
            # 4. 生成文件名
            file_ext = os.path.splitext(f_path)[1].lower()
            new_name = generate_readable_filename(description, file_ext, f_hash)
            
            # 5. 保存文件 - 使用修正后的分类
            dest_dir = os.path.join(TARGET_DIR, initial_cat)
            os.makedirs(dest_dir, exist_ok=True)
            final_path = os.path.join(dest_dir, new_name)
            
            # 避免文件名冲突
            counter = 1
            while os.path.exists(final_path) and counter < 100:
                name_body = new_name.rsplit('.', 1)[0]
                name_ext = new_name.rsplit('.', 1)[1] if '.' in new_name else ''
                new_name = f"{name_body}-{counter}.{name_ext}" if name_ext else f"{name_body}-{counter}"
                final_path = os.path.join(dest_dir, new_name)
                counter += 1
            
            shutil.copy2(f_path, final_path)
            
            # 6. 计算相对路径
            relative_path = os.path.relpath(final_path, TARGET_DIR).replace('\\', '/')
            
            # 7. 记录到数据库
            db_data.append({
                "md5": f_hash,
                "filename": new_name,
                "full_path": final_path,
                "relative_path": relative_path,
                "category": initial_cat,
                "original_category": original_cat_for_stats,  # 保留原始分类
                "classification_confidence": float(classification_confidence),
                "tags_cn": tags_cn,
                "tags_en": tags_en,
                "description": description,
                "original_filename": original_filename,  # 保存原始文件名
                "filename_keywords": extract_filename_keywords(original_filename),  # 保存提取的关键词
                "original_path": f_path,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            processed_hashes.add(f_hash)
            processed_count += 1
            
            # 8. 保存进度
            if processed_count % 5 == 0:
                with open(JSON_DB_PATH, 'w', encoding='utf-8') as f:
                    json.dump(db_data, f, ensure_ascii=False, indent=2)
                print(f"\n💾 已保存进度: {processed_count} 个文件")
            
            # 9. 打印详细信息
            print(f"\n✅ 处理成功: {original_filename}")
            print(f"   最终分类: {initial_cat} (置信度: {classification_confidence:.2f})")
            print(f"   原始分类: {original_cat_for_stats}")
            print(f"   场景描述: {description}")
            print(f"   中文标签: {', '.join(tags_cn)}")
            print(f"   英文标签: {', '.join(tags_en)}")
            print(f"   相对路径: {relative_path}")
            print(f"   保存为: {new_name}")
            
        except Exception as e:
            print(f"\n❌ 处理失败 {os.path.basename(f_path)}: {str(e)}")
            # 错误文件也放入未分类目录
            try:
                # 生成回退文件名
                file_ext = os.path.splitext(f_path)[1].lower()
                file_hash = get_file_md5(f_path)
                new_name = f"error_{file_hash[:6]}{file_ext}"
                
                # 保存到未分类目录
                dest_dir = os.path.join(TARGET_DIR, "未分类素材")
                os.makedirs(dest_dir, exist_ok=True)
                final_path = os.path.join(dest_dir, new_name)
                
                shutil.copy2(f_path, final_path)
                
                # 计算相对路径
                relative_path = os.path.relpath(final_path, TARGET_DIR).replace('\\', '/')
                
                # 记录到数据库
                db_data.append({
                    "md5": file_hash,
                    "filename": new_name,
                    "full_path": final_path,
                    "relative_path": relative_path,
                    "category": "未分类素材",
                    "classification_confidence": 0.0,
                    "tags_cn": ["错误", "处理失败"],
                    "tags_en": ["error", "processing_failed"],
                    "description": f"处理失败: {str(e)}",
                    "original_filename": os.path.basename(f_path),
                    "original_path": f_path,
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                processed_hashes.add(file_hash)
                processed_count += 1
                
                print(f"  🛡️ 错误文件已保存到: 未分类素材/{new_name}")
            except Exception as backup_e:
                print(f"  ❌ 错误处理也失败: {str(backup_e)}")
    
    # 最终保存
    with open(JSON_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db_data, f, ensure_ascii=False, indent=2)
    
    # 任务摘要
    print(f"\n{'='*60}")
    print("🎉 任务完成摘要")
    print(f"{'='*60}")
    print(f"📁 源目录: {SOURCE_DIRS}")
    print(f"📂 目标目录: {TARGET_DIR}")
    print(f"✅ 新增处理: {processed_count} 个文件")
    print(f"📊 总记录数: {len(db_data)}")
    
    # 分类统计
    category_stats = {}
    low_conf_stats = {}
    for item in db_data:
        category_stats[item["category"]] = category_stats.get(item["category"], 0) + 1
        
        # 低置信度统计
        if item["classification_confidence"] < 0.5:
            low_conf_stats[item["category"]] = low_conf_stats.get(item["category"], 0) + 1
    
    print("\n📈 分类统计:")
    for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        low_count = low_conf_stats.get(cat, 0)
        if low_count > 0:
            print(f"   {cat}: {count} 个文件 ({low_count} 个低置信度)")
        else:
            print(f"   {cat}: {count} 个文件")
    
    # 置信度统计
    avg_confidence = sum(item["classification_confidence"] for item in db_data) / len(db_data) if db_data else 0
    high_conf_files = [f for f in db_data if f["classification_confidence"] > 0.8]
    medium_conf_files = [f for f in db_data if 0.6 <= f["classification_confidence"] <= 0.8]
    low_conf_files = [f for f in db_data if f["classification_confidence"] < 0.6]
    
    print(f"\n📊 置信度统计:")
    print(f"   平均置信度: {avg_confidence:.3f}")
    print(f"   高置信度(>0.8): {len(high_conf_files)} 个文件 ({len(high_conf_files)/len(db_data)*100:.1f}%)")
    print(f"   中置信度(0.6-0.8): {len(medium_conf_files)} 个文件 ({len(medium_conf_files)/len(db_data)*100:.1f}%)")
    print(f"   低置信度(<0.6): {len(low_conf_files)} 个文件 ({len(low_conf_files)/len(db_data)*100:.1f}%)")
    print(f"   未分类素材目录: {category_stats.get('未分类素材', 0)} 个文件")
    
    # 总耗时计算
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    
    # 转换为分秒格式
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    
    # 格式化耗时字符串
    if minutes > 0:
        duration_str = f"{minutes}分{seconds}秒"
    else:
        duration_str = f"{seconds}秒"
    
    # 显示总耗时
    print(f"\n⏱️  总耗时: {duration_str} ({elapsed_seconds:.1f}秒)")
    print(f"⚡ 平均处理速度: {elapsed_seconds/max(processed_count,1):.2f}秒/文件")
    
    print(f"\n💡 智能融合提示: 原始文件名关键词与AI分析结果已平衡融合")
    print("✨ 系统资源已释放，5秒后自动退出...")
    time.sleep(5)

if __name__ == "__main__":
    main()