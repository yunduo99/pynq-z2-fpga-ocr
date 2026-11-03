# -*- coding: utf-8 -*-
"""
板卡端服务管理（start / stop / status / restart / log）

用法(在板卡 Jupyter 里):
    !cd /home/xilinx/jupyter_notebooks/pynq_project && python3 board_service.py status
    !cd /home/xilinx/jupyter_notebooks/pynq_project && python3 board_service.py start
    !cd /home/xilinx/jupyter_notebooks/pynq_project && python3 board_service.py stop
    !cd /home/xilinx/jupyter_notebooks/pynq_project && python3 board_service.py log
"""
import os, subprocess, sys, time, socket

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, 'board_camera_server.py')
LOG = '/tmp/cam_server.log'
PIDF = '/tmp/cam_server.pid'
PORT = 9999


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    return ((p.stdout or '') + (p.stderr or '')).strip()


def pids():
    out = sh("ps -eo pid,args | grep '[b]oard_camera_server.py'")
    res = []
    for line in out.split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            res.append(int(line.split()[0]))
        except Exception:
            pass
    return res


def port_open():
    try:
        s = socket.socket()
        s.settimeout(2)
        s.connect(('127.0.0.1', PORT))
        s.close()
        return True
    except Exception:
        return False


def cmd_status():
    p = pids()
    print("=" * 56)
    print("  板卡识别服务状态")
    print("=" * 56)
    if p:
        print(f"  ✅ 运行中  PID={p}")
        print(f"  端口 {PORT}: {'✅ 监听中' if port_open() else '❌ 未监听(可能正在初始化)'}")
        print(f"  日志: {LOG}")
        tail = sh(f"tail -6 {LOG}")
        print("  --- 最近日志 ---")
        for l in tail.split('\n'):
            print("   " + l[:120])
    else:
        print("  ❌ 未运行")
        print(f"  启动命令: python3 board_service.py start")
    # 内存
    mem = sh("free -m | sed -n 2p")
    print(f"  内存: {mem}")


def cmd_start():
    if pids():
        print("服务已在运行, 先停止旧的 ...")
        cmd_stop()
        time.sleep(2)
    print("启动识别服务 ...")
    # setsid 让服务完全脱离当前会话/进程组, 不会被 Jupyter 子进程结束时连带杀掉
    sh(f"cd {HERE} && setsid nohup python3 -u board_camera_server.py "
       f"> {LOG} 2>&1 < /dev/null & echo started")
    # 等就绪
    ok = False
    for i in range(40):
        time.sleep(1)
        if port_open():
            ok = True
            break
    print(f"  {'✅ 已就绪' if ok else '⚠️ 超时(看日志)'}  (等待 {i+1}s)")
    tail = sh(f"tail -4 {LOG}")
    for l in tail.split('\n'):
        print("   " + l[:120])


def cmd_stop():
    p = pids()
    if not p:
        print("服务未运行")
        return
    sh("pkill -f '[b]oard_camera_server.py'")
    time.sleep(2)
    print(f"  {'✅ 已停止' if not pids() else '⚠️ 仍在运行, 尝试强杀'}")
    if pids():
        sh("pkill -9 -f '[b]oard_camera_server.py'")
        time.sleep(1)
        print(f"  {'✅ 已强制停止' if not pids() else '❌ 停止失败'}")


def cmd_restart():
    cmd_stop()
    time.sleep(1)
    cmd_start()


def cmd_log():
    n = sys.argv[2] if len(sys.argv) > 2 else '40'
    print(sh(f"tail -{n} {LOG}"))


def main():
    act = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if act == 'start':
        cmd_start()
    elif act == 'stop':
        cmd_stop()
    elif act == 'restart':
        cmd_restart()
    elif act == 'log':
        cmd_log()
    elif act == 'status':
        cmd_status()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
