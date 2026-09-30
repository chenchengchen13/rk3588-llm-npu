// RKLLM benchmark: 加载计时 + prefill/decode tok/s 统计
// 用法: ./rkllm_bench <model.rkllm> <max_new_tokens> <max_context> <prompt> [runs]
#include <string.h>
#include <unistd.h>
#include <string>
#include "rkllm.h"
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <chrono>

using namespace std;

LLMHandle llmHandle = nullptr;
string g_text;
RKLLMPerfStat g_perf;
bool g_finished = false;

void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        LLMHandle _tmp = llmHandle;
        llmHandle = nullptr;
        rkllm_destroy(_tmp);
    }
    exit(signal);
}

int callback(RKLLMResult *result, void *userdata, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH)
    {
        g_perf = result->perf;
        g_finished = true;
    }
    else if (state == RKLLM_RUN_ERROR)
    {
        printf("\\run error\n");
        g_finished = true;
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        g_text += result->text;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " model_path max_new_tokens max_context_len prompt [runs]\n";
        return 1;
    }
    int max_new_tokens = std::atoi(argv[2]);
    int max_context_len = std::atoi(argv[3]);
    std::string prompt = argv[4];
    int runs = (argc >= 6) ? std::atoi(argv[5]) : 1;

    signal(SIGINT, exit_handler);

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];
    param.top_k = 1;
    param.top_p = 0.95;
    param.temperature = 0.8;
    param.repeat_penalty = 1.1;
    param.frequency_penalty = 0.0;
    param.presence_penalty = 0.0;
    param.max_new_tokens = max_new_tokens;
    param.max_context_len = max_context_len;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 0;
    param.extend_param.embed_flash = 1;

    auto t0 = std::chrono::steady_clock::now();
    int ret = rkllm_init(&llmHandle, &param, callback);
    auto t1 = std::chrono::steady_clock::now();
    if (ret != 0)
    {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[LOAD] rkllm_init %.1f ms\n", load_ms);

    // 可选: args 6-8 = system prompt / prefix / postfix, 覆盖内置 chat template
    if (argc >= 9)
    {
        ret = rkllm_set_chat_template(llmHandle, argv[6], argv[7], argv[8]);
        printf("[TEMPLATE] set_chat_template ret=%d system='%s' prefix='%s' postfix='%s'\n",
               ret, argv[6], argv[7], argv[8]);
    }

    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = 0;

    for (int r = 0; r < runs; r++)
    {
        g_text.clear();
        g_finished = false;
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.role = "user";
        rkllm_input.prompt_input = (char *)prompt.c_str();

        ret = rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
        if (ret != 0)
        {
            printf("rkllm_run failed\n");
            break;
        }
        if (r == 0)
        {
            printf("[TEXT]\n%s\n[/TEXT]\n", g_text.c_str());
        }
        float prefill_tps = g_perf.prefill_time_ms > 0 ? g_perf.prefill_tokens / (g_perf.prefill_time_ms / 1000.0f) : 0;
        float gen_tps = g_perf.generate_time_ms > 0 ? g_perf.generate_tokens / (g_perf.generate_time_ms / 1000.0f) : 0;
        printf("[PERF run %d] prefill %d tokens %.1f ms (%.2f tok/s) | decode %d tokens %.1f ms (%.2f tok/s) | mem %.1f MB\n",
               r, g_perf.prefill_tokens, g_perf.prefill_time_ms, prefill_tps,
               g_perf.generate_tokens, g_perf.generate_time_ms, gen_tps, g_perf.memory_usage_mb);
        // 清 KV cache 保证每次独立
        rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
    }

    exit_handler(0);
    return 0;
}
