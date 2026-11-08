# 许可证与资源范围

本项目不是“整个下载包内所有内容均为 MIT”。不同内容适用的范围如下。

| 内容 | 适用说明 |
| --- | --- |
| 本项目自有源码与脚本 | 采用根目录 [LICENSE](LICENSE) 中的 MIT 许可证；保留已引入代码的原作者声明。 |
| MiaCode 检测参考数据 | `src/maichart/generation/muri/reference/` 中的数据保留上游非商业分发范围，不改授为 MIT。详见该目录的 [LICENSE_SCOPE.md](src/maichart/generation/muri/reference/LICENSE_SCOPE.md) 及 [第三方声明](THIRD_PARTY_NOTICES.md)。包含这些数据的 wheel 也适用此区别。 |
| MERT 音乐编码器权重 | 默认 `m-a-p/MERT-v1-330M` 单独下载，其[官方模型卡](https://huggingface.co/m-a-p/MERT-v1-330M)声明 CC BY-NC 4.0；项目源码许可证不覆盖它。更换本地编码器时，按相应权重的许可执行。 |
| MiaView 预览播放器 | 单独下载；当前核对的播放器产物也注明资源与发布包的非商业使用／分发范围。保留并遵循下载产物中的 `LICENSE`、`LICENSE_SCOPE.md` 和 `THIRD_PARTY_NOTICES.md`；源码、素材和依赖的范围应分别查看，不以本项目 MIT 许可证替代。预览区提供许可说明入口。 |
| Composer 生成模型检查点 | 不随源码提供，源码的 MIT 许可证不自动覆盖检查点。发布权重时须另附模型说明与许可证，并核对训练素材及相关依赖的使用、分发范围。 |
| 用户歌曲、谱面与生成结果 | 不因使用本项目就自动取得 MIT 许可；音乐、素材和谱面的既有权利仍需分别处理。 |

完整来源、上游作者声明和例外见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
