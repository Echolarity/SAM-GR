#!/usr/bin/env bash
# =============================================================================
# 任务管理器: 后台运行 main.sh (start / stop / restart / status / logs)
#   用法: bash run.sh [start|stop|restart|status|logs]
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$HERE/logs"
PIDFILE="$HERE/logs/main.pid"
mkdir -p "$LOGDIR"

ACTION="${1:-status}"

running_pid() {
    if [ -f "$PIDFILE" ]; then
        local pid
        pid="$(cat "$PIDFILE" 2>/dev/null || true)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
        fi
    fi
}

case "$ACTION" in
    start)
        if [ -n "$(running_pid)" ]; then
            echo "已在运行 (pid=$(running_pid))"
            exit 0
        fi
        nohup bash "$HERE/main.sh" > "$LOGDIR/main.log" 2>&1 &
        echo $! > "$PIDFILE"
        echo "已启动 (pid=$!) 日志: $LOGDIR/main.log"
        ;;
    stop)
        pid="$(running_pid)"
        if [ -n "$pid" ]; then
            # 按进程组终止 (连同 torch/deepspeed 子进程)
            kill -- -"$(ps -o pgid= -p "$pid" | tr -d ' ')" 2>/dev/null || kill "$pid" 2>/dev/null
            pkill -P "$pid" 2>/dev/null || true
            echo "已停止 (pid=$pid)"
        else
            echo "未在运行"
        fi
        rm -f "$PIDFILE"
        ;;
    restart)
        bash "$0" stop
        sleep 2
        bash "$0" start
        ;;
    status)
        pid="$(running_pid)"
        if [ -n "$pid" ]; then
            echo "运行中 (pid=$pid)"
        else
            echo "未在运行"
        fi
        ;;
    logs)
        tail -n 100 "$LOGDIR/main.log" 2>/dev/null || echo "无日志"
        ;;
    *)
        echo "用法: bash run.sh [start|stop|restart|status|logs]"
        exit 1
        ;;
esac
