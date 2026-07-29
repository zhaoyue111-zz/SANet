import time


# 正在debug中，需要持续占用几张卡
while True:
    try:
        # 程序将永远睡眠
        time.sleep(1000000000)  # 这里的参数表示睡眠的秒数，你可以根据需要调整
    except KeyboardInterrupt:
        # 可以处理键盘中断，让程序在用户按下 Ctrl+C 时退出
        print("程序被中断，即将退出")
        break