# DEX: یادگیری تقویتی هدایت‌شده با نمایش همراه با کاوش کارآمد برای خودکارسازی وظایف ربات جراحی

این پیاده‌سازی رسمی PyTorch از مقاله "**یادگیری تقویتی هدایت‌شده با نمایش همراه با کاوش کارآمد برای خودکارسازی وظایف ربات جراحی**" (ICRA 2023) است.  
<p align="left"> 
  <img width="98%" src="docs/resources/dex_teaser.png">
</p>

# پیش‌نیازها
* Ubuntu 18.04  
* Python 3.7+  

# دستورالعمل نصب

1. این مخزن را کلون کنید:
```bash
git clone https://github.com/fatemeahmadpour/DEX-Modified.git
cd DEX
git clone https://github.com/fatemeahmadpour/SurRoL.git SurRoL 
```

2. یک محیط مجازی ایجاد کنید:
```bash
conda create -n dex python=3.8
conda activate dex
```

3. بسته‌ها را نصب کنید:
```bash
pip3 install -e SurRoL/	# نصب محیط‌های surrol
pip3 install -r requirements.txt
pip3 install -e .
```

4. سپس یک خط کد به بالای فایل `gym/gym/envs/__init__.py` اضافه کنید تا وظایف SurRoL ثبت شوند:
```python
# مسیر: anaconda3/envs/dex/lib/python3.8/site-packages/
import surrol.gym
```

# استفاده
دستورات برای اجرای DEX و تمامی الگوریتم‌های پایه. نتایج در WandB ثبت خواهند شد. پیش از اجرای دستورات زیر، لطفاً موجودیت wandb را در فایل [```train.yaml```](dex/configs/train.yaml#L26) به حساب خود تغییر دهید.

ما داده‌های نمایش را از طریق کنترل‌کننده‌های اسکریپت‌شده ارائه‌شده توسط SurRoL جمع‌آوری می‌کنیم. به عنوان مثال، وظیفه NeedlePick:  
```bash
mkdir SurRoL/surrol/data/demo
python SurRoL/surrol/data/data_generation.py --env NeedlePick-v0 
```

## ساخت دیتا برای مدیریت نخ بخیه (محیط ساختگی من)
```bash
python SurRoL/surrol/data/data_generation.py --env SutureThreadManagement-v0 
```

### آموزش اجرا **DEX** برای مدیریت نخ بخیه با الگوریتم های مختلف(محیط ساختگی من):
```bash
python3 train.py task=SutureThreadManagement-v0 agent=dex use_wb=True
```

### و به همین ترتیب...


- آموزش **DEX**:
```bash
python3 train.py task=NeedlePick-v0 agent=dex use_wb=True
```

- آموزش **SAC**:
```bash
python3 train.py task=NeedlePick-v0 agent=sac use_wb=True
```

- آموزش **DDPG**:
```bash
python3 train.py task=NeedlePick-v0 agent=ddpg use_wb=True
```

- آموزش **DDPGBC**:
```bash
python3 train.py task=NeedlePick-v0 agent=ddpgbc use_wb=True
```

- آموزش **CoL**:
```bash
python3 train.py task=NeedlePick-v0 agent=col use_wb=True
```

- آموزش **AMP**:
```bash
python3 train.py task=NeedlePick-v0 agent=amp use_wb=True
```

- آموزش **AWAC**:
```bash
python3 train.py task=NeedlePick-v0 agent=awac use_wb=True
```

- آموزش **SQIL**:
```bash
python3 train.py task=NeedlePick-v0 agent=sqil use_wb=True
```

تمامی دستورات را می‌توان برای دیگر وظایف جراحی با جایگزینی NeedlePick با محیط مربوطه در دستورات اجرا کرد (چه برای جمع‌آوری داده‌ها و چه برای آموزش RL).

ما همچنین اجرای موازی همگام‌سازی‌شده برای آموزش RL را پیاده‌سازی کرده‌ایم، مثلاً اجرای ۴ فرآیند آموزش موازی:
```bash
mpirun -np 4 python -m train agent=dex task=NeedlePick-v0 use_wb=True
```

توجه داشته باشید که آموزش موازی می‌تواند منجر به عملکرد ناپایدار شود که نیاز به تنظیم پارامترهای ابر دارد.

## دستورات ارزیابی
اسکریپتی نیز برای ارزیابی مدل ذخیره‌شده فراهم شده است. مسیر مدل مورد ارزیابی باید در فایل تنظیمات [```eval.yaml```](dex/configs/eval.yaml) مشخص شود، جایی که نقطه بازیابی توسط `ckpt_episode` تعیین می‌شود. به عنوان مثال:  
- ارزیابی مدل آموزش‌دیده توسط **DEX** در NeedlePick-v0:
```bash
python3 eval.py task=NeedlePick-v0 agent=dex ckpt_episode=latest
```

# شروع به ویرایش کد
## ویرایش پارامترهای ابر
پارامترهای پیش‌فرض در `dex/configs` تعریف شده‌اند، جایی که [```train.yaml```](dex/configs/train.yaml) تنظیمات آزمایش را تعریف می‌کند و فایل‌های YAML در مسیر [```agent```](dex/configs/agent) پارامترهای هر روش را تعریف می‌کنند. تغییرات این پارامترها را می‌توان به‌طور مستقیم در فایل‌های تنظیمات آزمایش یا عامل تعریف کرد یا از طریق دستور ترمینال منتقل کرد. برای مثال:  
```bash
python3 train.py task=NeedleRegrasp-v0 agent=dex use_wb=True batch_size=256 agent.aux_weight=10
```

## افزودن یک الگوریتم RL جدید
الگوریتم‌های اصلی RL در کلاس `BaseAgent` پیاده‌سازی شده‌اند. برای افزودن یک الگوریتم جدید، یک فایل جدید باید در مسیر `dex/agents` ایجاد شود و [```BaseAgent```](dex/agents/base.py#L8) باید زیرکلاس شود. به‌طور خاص، هر شبکه مورد نیاز (actor, critic و غیره) باید ساخته شود و توابع `update(...)` و `get_action(...)` بازنویسی شوند. برای مثال، به پیاده‌سازی DDPGBC در [```DDPGBC```](dex/agents/ddpgbc.py#L7) مراجعه کنید. پس از تکمیل پیاده‌سازی، نیاز به ثبت در [```factory.py```](dex/agents/factory.py) و ایجاد یک فایل تنظیمات در مسیر [```agent```](dex/configs/agent) برای مشخص کردن پارامترهای مدل است.

## انتقال به پلتفرم شبیه‌سازی دیگر
کد ما برای محیط‌های استاندارد مبتنی بر OpenAI gym با هدف هدف‌مند طراحی شده است و می‌تواند به‌راحتی به پلتفرم‌های دیگر انتقال یابد در صورتی که رابط‌های مشابه ارائه شوند. اگر چنین رابطی ارائه نشود، باید تغییراتی اعمال شود تا سازگار شود، مانند بافر بازپخش و ابزارهای نمونه‌برداری. در آینده کد ما عمومی‌تر خواهد شد.

# مسیریابی کد

```
dex
  |- agents                # پیاده‌سازی الگوریتم‌های اصلی در کلاس‌های عامل
  |- components            # زیرساخت‌های قابل استفاده مجدد برای آموزش مدل
  |    |- checkpointer.py  # مدیریت ذخیره‌سازی و بارگذاری مدل
  |    |- normalizer.py    # نرمال‌سازی برای ورودی‌های برداری
  |    |- logger.py        # پیاده‌سازی عملکرد ثبت‌کننده اصلی با استفاده از wandB
  |
  |- configs               # تنظیمات آزمایش
  |    |- train.yaml       # تنظیمات برای آموزش RL
  |    |- eval.yaml        # تنظیمات برای ارزیابی RL
  |    |- agent            # تنظیمات برای هر الگوریتم (dex، ddpg، ddpgbc و غیره)
  |
  |- modules               # اجزای معماری قابل استفاده مجدد
  |    |- critic.py        # پیاده‌سازی‌های اصلی critic (مثلاً critic مبتنی بر MLP)
  |    |- distributions.py # ابزارهای توزیع PyTorch برای مدل چگالی
  |    |- policy.py    	   # پیاده‌سازی‌های اصلی actor
  |    |- replay_buffer.py # بافر بازپخش HER با استراتژی نمونه‌برداری آینده
  |    |- sampler.py       # نمونه‌بردار rollout برای جمع‌آوری تجربه
  |    |- subnetworks.py   # شبکه‌های اصلی
  |
  |- trainers              # اسکریپت اصلی آموزش مدل، ساخت همه اجزا و اجرای حلقه آموزش و ثبت لاگ
  |
  |- utils                 # ابزارهای عمومی و RL، ابزارهای PyTorch / مصورسازی و غیره
  |- train.py              # راه‌انداز آزمایش
  |- eval.py               # راه‌انداز ارزیابی
```
