"""生成中文+推理校准数据集，供 rkllm-toolkit 量化校准用。
输出: data_quant_zh.json （约 140 条 {input, target}）"""
import json
import os
import random

random.seed(42)
samples = []

# 1) 中文数学应用题 + 逐步推理（60 条，覆盖加减乘除混合）
items = [("苹果", "袋"), ("铅笔", "盒"), ("书", "包"), ("鸡蛋", "篮"), ("瓶子", "箱"), ("糖果", "包")]
for _ in range(60):
    name, unit = random.choice(items)
    a = random.randint(2, 15)
    b = random.randint(2, 9)
    c = random.randint(2, 12)
    style = random.randint(0, 2)
    if style == 0:
        q = f"小明有{a}个{name}，又买了{b}{unit}{name}每{unit}{c}个，他现在一共有多少个{name}？请逐步推理。"
        t = (f"首先，小明原本有{a}个{name}。\n\n接着，他买了{b}{unit}{name}，每{unit}有{c}个{name}，"
             f"所以买了的{name}数量是{b}乘以{c}，等于{b*c}个{name}。\n\n"
             f"最后，将原本的{a}个和购买的{b*c}个相加，{a}加{b*c}等于{a+b*c}。\n\n"
             f"所以，小明现在一共有{a+b*c}个{name}。")
    elif style == 1:
        total = a * b
        q = f"一箱有{a}个{name}，{b}箱一共有多少个{name}？"
        t = f"每箱有{a}个{name}，共有{b}箱，所以总数是{a}乘以{b}，等于{total}个{name}。"
    else:
        total = a * c
        q = f"老师买了{total}个{name}，平均分给{a}个同学，每个同学分到多少个？"
        t = f"把{total}个{name}平均分成{a}份，每份是{total}除以{a}，等于{c}个。所以每个同学分到{c}个{name}。"
    samples.append({"input": q, "target": t})

# 2) 中文常识/知识问答（30 条）
qa = [
    ("中国的首都是哪里？", "中国的首都是北京。"),
    ("水的化学式是什么？", "水的化学式是H2O，由两个氢原子和一个氧原子组成。"),
    ("地球绕太阳公转一周大约需要多长时间？", "地球绕太阳公转一周大约需要365.25天，即一年。"),
    ("请用一句话介绍北京理工大学。", "北京理工大学是隶属于工业和信息化部的全国重点大学，是国家“双一流”建设高校，以理工为主，工理管文协调发展。"),
    ("什么是人工智能？", "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门技术科学。"),
    ("光速大约是多少？", "真空中的光速约为每秒299792458米，约等于每秒30万公里。"),
    ("唐朝的开国皇帝是谁？", "唐朝的开国皇帝是李渊，他于618年建立唐朝，定都长安。"),
    ("什么是光合作用？", "光合作用是绿色植物利用光能，把二氧化碳和水合成有机物并释放氧气的过程。"),
    ("一年有多少个节气？", "一年有二十四个节气，起源于黄河流域，是中国古代订立的一种用来指导农事的补充历法。"),
    ("计算机中1KB等于多少字节？", "1KB等于1024字节。"),
    ("请解释什么是勾股定理。", "勾股定理指直角三角形的两条直角边的平方和等于斜边的平方，即a²+b²=c²。"),
    ("长江有多长？", "长江全长约6300公里，是中国第一长河，也是亚洲第一长河。"),
    ("什么是机器学习？", "机器学习是人工智能的一个分支，通过让计算机从数据中学习规律，从而对未知数据进行预测或决策。"),
    ("春节通常在什么时候？", "春节是农历正月初一，通常在公历1月下旬至2月中旬之间。"),
    ("人体有多少块骨骼？", "成年人通常有206块骨骼。"),
    ("什么是量子计算？", "量子计算是利用量子力学原理（如叠加和纠缠）进行信息处理的一种计算模式，在某些问题上可比经典计算快得多。"),
    ("黄河发源于哪里？", "黄河发源于青藏高原的巴颜喀拉山脉，最终注入渤海。"),
    ("声音在空气中的传播速度大约是多少？", "声音在15摄氏度的空气中传播速度约为每秒340米。"),
    ("什么是操作系统？", "操作系统是管理计算机硬件与软件资源的系统软件，常见的有Windows、Linux和macOS。"),
    ("北京奥运会是哪一年举办的？", "北京夏季奥运会于2008年举办，北京还于2022年举办了冬季奥运会。"),
    ("什么是碳中和？", "碳中和指通过节能减排、植树造林等方式，抵消自身产生的二氧化碳排放，实现净零排放。"),
    ("《红楼梦》的作者是谁？", "《红楼梦》的作者是清代作家曹雪芹，后四十回一般认为由高鹗续写。"),
    ("什么是DNA？", "DNA即脱氧核糖核酸，是携带遗传信息的分子，由四种碱基组成双螺旋结构。"),
    ("太阳系有几大行星？", "太阳系有八大行星，分别是水星、金星、地球、火星、木星、土星、天王星和海王星。"),
    ("什么是5G？", "5G是第五代移动通信技术，具有高速率、低时延、大连接的特点。"),
    ("三角形内角和是多少度？", "平面几何中，三角形的内角和是180度。"),
    ("什么是通货膨胀？", "通货膨胀指货币供给过多导致物价持续上涨、货币购买力下降的经济现象。"),
    ("鲁迅的代表作有哪些？", "鲁迅的代表作有《呐喊》《彷徨》《朝花夕拾》等，其中《狂人日记》是中国现代文学史上第一篇白话小说。"),
    ("什么是区块链？", "区块链是一种分布式账本技术，通过密码学方法将数据块按时间顺序链接，具有去中心化和不可篡改的特点。"),
    ("珠穆朗玛峰有多高？", "珠穆朗玛峰最新测量高程为8848.86米，是世界最高峰。"),
]
for q, t in qa:
    samples.append({"input": q, "target": t})

# 3) 英文推理（20 条，贴合 R1 蒸馏模型的训练分布）
en = [
    ("What is 15 plus 27?", "To find the sum, add 15 and 27: 15 + 27 = 42. So the answer is 42."),
    ("If a train travels 60 km per hour for 3 hours, how far does it go?", "Distance equals speed multiplied by time: 60 * 3 = 180 km. The train travels 180 kilometers."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Explain what a neural network is in one sentence.", "A neural network is a computational model inspired by the brain, composed of layers of interconnected nodes that learn patterns from data."),
    ("What is 8 multiplied by 7?", "8 multiplied by 7 equals 56."),
    ("If you have 24 candies and share them equally among 6 friends, how many does each get?", "Divide 24 by 6: 24 / 6 = 4. Each friend gets 4 candies."),
    ("What is the boiling point of water at sea level?", "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level."),
    ("What is the square root of 144?", "The square root of 144 is 12, because 12 times 12 equals 144."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
    ("What is 100 divided by 4?", "100 divided by 4 equals 25."),
    ("Explain photosynthesis briefly.", "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen."),
    ("What is the largest planet in the solar system?", "Jupiter is the largest planet in the solar system."),
    ("If a book costs 12 yuan and you pay with 50 yuan, how much change do you get?", "Subtract 12 from 50: 50 - 12 = 38. You get 38 yuan in change."),
    ("What is 9 squared?", "9 squared is 81."),
    ("Name three primary colors.", "The three primary colors are red, yellow, and blue."),
    ("What does CPU stand for?", "CPU stands for Central Processing Unit, the main processor of a computer."),
    ("If x + 5 = 12, what is x?", "Subtract 5 from both sides: x = 12 - 5 = 7."),
    ("What is the freezing point of water?", "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit) at standard atmospheric pressure."),
    ("How many minutes are in two hours?", "Each hour has 60 minutes, so two hours have 2 * 60 = 120 minutes."),
    ("What is gravity?", "Gravity is the force of attraction between objects with mass; on Earth it pulls objects toward the ground at about 9.8 m/s²."),
]
for q, t in en:
    samples.append({"input": q, "target": t})

# 4) 中文指令跟随/开放生成（30 条）
gen = [
    ("请写一句关于春天的诗。", "春风又绿江南岸，明月何时照我还。"),
    ("用三句话总结为什么要多读书。", "读书能开阔视野，让我们看到更大的世界。读书能训练思维，提升分析问题的能力。读书能丰富内心，让人生更加充实。"),
    ("请给出一个保持健康生活的建议。", "保持规律作息、均衡饮食和适量运动，是维持身体健康最重要的三个习惯。"),
    ("如何把大象放进冰箱？分三步回答。", "第一步，打开冰箱门。第二步，把大象放进去。第三步，关上冰箱门。"),
    ("请解释“亡羊补牢”这个成语的意思。", "亡羊补牢意思是羊丢失之后再去修补羊圈，比喻出了问题以后及时补救，还不算晚。"),
    ("写一句鼓励备考学生的话。", "每一份努力都不会被辜负，坚持下去，你终会抵达想去的远方。"),
    ("请列举三种清洁能源。", "太阳能、风能和水能是三种常见的清洁能源。"),
    ("什么是“因材施教”？", "因材施教指根据学生的不同特点和水平，采用不同的教育方法，使每个人都能得到适合的发展。"),
    ("请用简单的话解释什么是递归。", "递归是一种解决问题的方法：函数在执行过程中调用自身，把大问题不断拆成相同结构的小问题，直到最小情况直接求解。"),
    ("描述一下雨后的空气。", "雨后的空气格外清新，带着泥土和青草的湿润气息，让人忍不住深呼吸。"),
    ("请给出学习英语的两条建议。", "第一，坚持每天接触英语，比如阅读和听力练习。第二，大胆开口说，不怕犯错，在实践中进步。"),
    ("什么是“塞翁失马，焉知非福”？", "这个成语比喻坏事在一定条件下可能变成好事，提醒人们用辩证的眼光看待得失。"),
    ("请写一段自我介绍，身份是大学生。", "大家好，我是一名大学生，主修电子信息。我热爱技术，喜欢动手实践，希望未来能把所学应用到真实的工程问题中。"),
    ("如何缓解学习压力？", "可以通过运动、听音乐、与朋友交流等方式缓解压力，同时合理安排作息，保证充足睡眠。"),
    ("请解释什么是惯性。", "惯性是物体保持原有运动状态的性质，质量越大，惯性越大。"),
    ("用一句话描述秋天。", "秋天是收获的季节，金黄的落叶和凉爽的微风让人心旷神怡。"),
    ("什么是团队合作？", "团队合作是成员为了共同目标分工协作、相互支持、共同承担责任的过程。"),
    ("请推荐一本你读过的书并说明理由。", "我推荐《活着》，它用朴素的语言讲述了普通人的一生，让人重新思考生命的意义。"),
    ("如何提高编程能力？", "多写代码、多读优秀源码、多做项目，并且养成调试和总结的习惯，编程能力就会稳步提升。"),
    ("请解释什么是浮力。", "浮力是流体对浸入其中的物体施加的向上的力，大小等于物体排开流体的重量，这就是阿基米德原理。"),
    ("写一句关于时间宝贵的名言警句。", "一寸光阴一寸金，寸金难买寸光阴。"),
    ("什么是摩擦力？", "摩擦力是两个相互接触的物体在相对运动或有相对运动趋势时，在接触面上产生的阻碍运动的力。"),
    ("请描述一次难忘的旅行。", "去年夏天我去了海边，清晨的日出染红了整片天空，海浪拍打礁石的声音至今让我难忘。"),
    ("什么是强迫症？用通俗的话解释。", "强迫症是一种心理状态，表现为反复出现不必要的想法或行为，明知道没必要却难以控制。"),
    ("请给出节约用水的三条建议。", "随手关紧水龙头；淘米水可以用来浇花；缩短淋浴时间。"),
    ("什么是人工智能中的过拟合？", "过拟合指模型在训练数据上表现很好，但在新数据上表现差，说明模型死记硬背了训练数据而没有学到通用规律。"),
    ("请用比喻的方式解释什么是电流。", "电流就像水管里的水流：电压是水压，电阻是水管的粗细和阻力，电流就是流动的水。"),
    ("如何准备一场重要的面试？", "提前了解公司和岗位，梳理自己的项目经历并准备量化成果，模拟常见问题进行练习，面试当天保持自信和真诚。"),
    ("请解释什么是回声。", "回声是声音传播遇到障碍物反射回来形成的，当反射声与原声间隔超过0.1秒时，人耳就能分辨出回声。"),
    ("写一句毕业赠言。", "愿此去前程似锦，再相逢依旧如故。"),
]
for q, t in gen:
    samples.append({"input": q, "target": t})

out = os.path.expanduser("~/rkllm_dl/data_quant_zh.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=1)
print(f"written {len(samples)} samples -> {out}")