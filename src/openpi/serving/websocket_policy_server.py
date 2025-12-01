import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy   # 一个策略对象，_base_policy.BasePolicy 的实例，包含 infer 方法
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        # 阻塞运行，启动服务器，启动异步事件循环
        asyncio.run(self.run())

    async def run(self):
        # 启动 WebSocket 服务器
        async with _server.serve(
            self._handler,  # 处理每个客户端连接的回调函数
            self._host,
            self._port,
            compression=None,   # 禁用 WebSocket 压缩
            max_size=None, # 不限制消息大小
            process_request=_health_check, # 健康检查
        ) as server:
            await server.serve_forever() # 保持服务器运行

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # 客户端连接后，首先发送 metadata 信息
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                # 异步接收数据， 数据解包成 Python 对象
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                # 调用策略对象的 infer 方法生成动作，记录推理耗时
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                # 把服务器端计时信息加入返回动作
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                # 包含总耗时
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                # 发送动作给客户端,更新 prev_total_time 以便下一次记录
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    # 如果请求路径是 /healthz，返回 HTTP 200;否则继续 WebSocket 的正常处理
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
