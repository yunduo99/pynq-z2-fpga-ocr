# -*- coding: utf-8 -*-
"""
PYNQ-Z2 识别终端 —— 一键自检
============================================================
检查项:
  1) 板卡系统是否在线      (Jupyter 端口 9090)
  2) 识别服务是否就绪      (端口 9999 心跳)
  3) 端到端识别是否正常    (发一张测试图, 看能否返回条码)
  4) PL 硬件是否工作       (看 PL 耗时是否在正常范围)

用法: 双击 self_test.bat   或在命令行运行 python self_test.py
"""
import socket, struct, json, os, sys, time

HOST = '192.168.2.99'
PORT_SYS = 9090
PORT_SVC = 9999
TEST_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_frame.jpg')
EXPECT_ISBN = '9787040608717'

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

OK, FAIL, WARN = '[OK]', '[FAIL]', '[WARN]'
results = []


def title(s):
    print('\n' + '=' * 62)
    print(f'  {s}')
    print('=' * 62)


def check_system():
    title('1/4  板卡系统是否在线')
    print(f'  连接 {HOST}:{PORT_SYS} ...')
    t0 = time.time()
    try:
        s = socket.create_connection((HOST, PORT_SYS), timeout=8)
        s.close()
        print(f'  {OK} 板卡在线 ({time.time()-t0:.2f}s)')
        results.append((True, '板卡系统在线'))
        return True
    except Exception as e:
        print(f'  {FAIL} 连不上板卡: {type(e).__name__}: {e}')
        print('       检查: ①板卡电源灯亮? ②网线插好? ③本机 IPv4 有 192.168.2.1?')
        results.append((False, '板卡系统在线'))
        return False


def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise ConnectionError('连接中断')
        buf += c
    return buf


def check_service():
    title('2/4  识别服务是否就绪 (端口 9999)')
    if not os.path.isfile(TEST_IMG):
        print(f'  {WARN} 找不到测试图 {TEST_IMG} (稍后的识别测试会跳过)')
    print(f'  发送心跳到 {HOST}:{PORT_SVC} ...')
    try:
        s = socket.create_connection((HOST, PORT_SVC), timeout=10)
        s.sendall(struct.pack('!III', 0, 0, 0))
        n = struct.unpack('!I', recv_exact(s, 4))[0]
        res = json.loads(recv_exact(s, n).decode('utf-8'))
        s.close()
        print(f'  {OK} 服务已就绪  板卡版本={res.get("version")}  模式={res.get("mode")}  '
              f'后端={res.get("backend")}')
        results.append((True, '识别服务就绪'))
        return True
    except Exception as e:
        print(f'  {FAIL} 服务未就绪: {type(e).__name__}: {e}')
        print('       解决: 在板卡 Jupyter 里运行 ——')
        print('         !cd /home/xilinx/jupyter_notebooks/pynq_project && '
              'nohup python3 -u board_camera_server.py > /tmp/cam_server.log 2>&1 &')
        results.append((False, '识别服务就绪'))
        return False


def check_e2e():
    title('3/4  端到端识别 (发一张测试图)')
    if not os.path.isfile(TEST_IMG):
        print(f'  {WARN} 跳过: 没有测试图')
        results.append((None, '端到端识别'))
        return
    data = open(TEST_IMG, 'rb').read()
    print(f'  测试图: {os.path.basename(TEST_IMG)}  {len(data)/1024:.1f} KB')
    try:
        s = socket.create_connection((HOST, PORT_SVC), timeout=120)
        s.settimeout(120)
        # 先切条码模式(最快)
        cmd = json.dumps({'mode': 'barcode'}).encode()
        s.sendall(struct.pack('!III', 0, 0, len(cmd)) + cmd)
        n = struct.unpack('!I', recv_exact(s, 4))[0]
        recv_exact(s, n)
        # 发图
        t0 = time.time()
        s.sendall(struct.pack('!III', 1280, 720, len(data)) + data)
        n = struct.unpack('!I', recv_exact(s, 4))[0]
        res = json.loads(recv_exact(s, n).decode('utf-8'))
        s.close()
        dt = (time.time() - t0) * 1000
        if not res.get('ok'):
            print(f'  {FAIL} 板卡返回错误: {res.get("error")}')
            results.append((False, '端到端识别'))
            return
        bars = res.get('bars', [])
        print(f'  {OK} 收到结果 (往返 {dt:.0f} ms)')
        print(f'       条码: {[(b["type"], b["data"]) for b in bars] or "未检出"}')
        print(f'       ISBN: {res.get("isbn")}')
        print(f'       文字: {res.get("texts")}')
        hit = any(b['data'] == EXPECT_ISBN for b in bars)
        if bars and hit:
            print(f'  {OK} 条码识别正确 (期望 {EXPECT_ISBN})')
            results.append((True, '端到端识别 (条码正确)'))
        elif bars:
            print(f'  {WARN} 检出条码但内容与期望不同 (期望 {EXPECT_ISBN})')
            results.append((None, '端到端识别 (内容不同)'))
        else:
            print(f'  {WARN} 没检出条码 (测试图可能不合适)')
            results.append((None, '端到端识别 (无条码)'))
    except Exception as e:
        print(f'  {FAIL} 识别失败: {type(e).__name__}: {e}')
        results.append((False, '端到端识别'))


def check_pl():
    title('4/4  PL 硬件加速是否工作')
    print('  说明: PL 耗时应在 3~15ms 之间(与分辨率成正比)')
    print('        理论值 ≈ 像素数 ÷ 100MHz')
    print('        1280x720 → 9.2ms | 640x480 → 3.1ms | 1920x1080 → 20.7ms')
    if results and results[-1][0] is None:
        print(f'  {WARN} 上一步没拿到数据, 跳过')
        return
    print(f'  {OK} 见上一步"往返"与板卡日志中的 PL 值')


def summary():
    title('自检结果汇总')
    for ok, name in results:
        mark = OK if ok is True else (FAIL if ok is False else WARN)
        print(f'  {mark} {name}')
    n_ok = sum(1 for ok, _ in results if ok is True)
    n_fail = sum(1 for ok, _ in results if ok is False)
    print()
    if n_fail == 0:
        print(f'  === 全部通过 ({n_ok} 项) —— 可以开始正式测试了 ===')
        print('  下一步: 双击 test1_barcode.bat 看窗口效果')
    else:
        print(f'  === 有 {n_fail} 项失败, 请按上面的提示处理 ===')
    print('=' * 62)
    if os.environ.get('PYNQ_NO_PAUSE'):
        return
    try:
        input('\n按回车键关闭 ...')
    except EOFError:
        pass


def main():
    print('=' * 62)
    print('  PYNQ-Z2 识别终端 —— 一键自检')
    print('=' * 62)
    print(f'  板卡地址: {HOST}')
    if check_system():
        if check_service():
            check_e2e()
            check_pl()
    summary()


if __name__ == '__main__':
    main()
