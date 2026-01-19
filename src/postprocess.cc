#include "yolov12.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>
#define LABEL_NALE_TXT_PATH "/data/github_project/YOLOv12_RK3588_object_detect/model/bird.txt"  // *****你需要在此处将路径改为你的txt的绝对路径

static char *labels[OBJ_CLASS_NUM];

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

static int loadLabelName(const char *locationFilename, char *label[])
{
    printf("load lable %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);
    return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

// 核心逻辑：解析单输出 Tensor
// 假设 Model Output Shape: [1, 16, 8400] 或者 [1, 8400, 16]
// 通常 Ultralytics 导出是 [1, 4+cls, 8400]。
// 我们在 C++ 中通过遍历 8400 个 anchor 来处理。
static int process_one_output(int8_t* buf, int32_t zp, float scale,
                              int anchor_num, int attr_num, int cls_num,
                              int stride_anchor, int stride_attr, // 用于控制内存访问步长
                              std::vector<float>& filterBoxes, 
                              std::vector<float>& objProbs, 
                              std::vector<int>& classId, 
                              float conf_threshold)
{
    int validCount = 0;
    
    // 遍历每一个 anchor (共 8400 个)
    for (int i = 0; i < anchor_num; i++) {
        // 计算当前 anchor 的数据起始位置
        // 如果是 [1, 16, 8400]，stride_anchor通常是1，stride_attr通常是8400
        // 如果是 [1, 8400, 16]，stride_anchor通常是16，stride_attr通常是1
        // 下面的调用处会决定这两个参数
        
        // 1. 查找最大类别概率
        float max_score = -1.0f;
        int max_class_id = -1;
        
        // 前4个是 box (cx, cy, w, h)，从第4个索引开始是 class scores
        for (int c = 0; c < cls_num; c++) {
            // 获取量化数值
            int8_t q_score = buf[i * stride_anchor + (4 + c) * stride_attr];
            // 反量化
            float score = deqnt_affine_to_f32(q_score, zp, scale);
            
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }

        // 2. 阈值筛选
        if (max_score > conf_threshold) {
            // 解析 Box (cx, cy, w, h)
            float cx = deqnt_affine_to_f32(buf[i * stride_anchor + 0 * stride_attr], zp, scale);
            float cy = deqnt_affine_to_f32(buf[i * stride_anchor + 1 * stride_attr], zp, scale);
            float w  = deqnt_affine_to_f32(buf[i * stride_anchor + 2 * stride_attr], zp, scale);
            float h  = deqnt_affine_to_f32(buf[i * stride_anchor + 3 * stride_attr], zp, scale);

            // 转换为 x1, y1, w, h (为了后续 NMS 方便，通常转为左上角坐标)
            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;

            filterBoxes.push_back(x1);
            filterBoxes.push_back(y1);
            filterBoxes.push_back(w);
            filterBoxes.push_back(h);

            objProbs.push_back(max_score);
            classId.push_back(max_class_id);
            validCount++;
        }
    }
    return validCount;
}

// 针对 FP32 模型的重载 (如果不想用量化，或者 outputs[0].want_float=1)
static int process_one_output_fp32(float* buf, 
                                   int anchor_num, int attr_num, int cls_num,
                                   int stride_anchor, int stride_attr,
                                   std::vector<float>& filterBoxes, 
                                   std::vector<float>& objProbs, 
                                   std::vector<int>& classId, 
                                   float conf_threshold)
{
    int validCount = 0;
    
    // 添加一个计数器，只打印前几个 anchor 的值用于调试
    int debug_print_count = 0;

    for (int i = 0; i < anchor_num; i++) 
    {
        float max_score = -1.0f;
        int max_class_id = -1;

        for (int c = 0; c < cls_num; c++) 
        {
            float score = buf[i * stride_anchor + (4 + c) * stride_attr];
            if (score > max_score) 
            {
                max_score = score;
                max_class_id = c;
            }
        }

        // [调试代码] 打印前 10 个 anchor 的最大分数，看看是不是正常的 0.xx 小数
        // if (debug_print_count < 10) 
        // {
        //     printf("Debug Anchor[%d]: max_score=%f, class=%d\n", i, max_score, max_class_id);
        //     debug_print_count++;
        // }

        if (max_score > conf_threshold) 
        {
            float cx = buf[i * stride_anchor + 0 * stride_attr];
            float cy = buf[i * stride_anchor + 1 * stride_attr];
            float w  = buf[i * stride_anchor + 2 * stride_attr];
            float h  = buf[i * stride_anchor + 3 * stride_attr];

            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;

            filterBoxes.push_back(x1);
            filterBoxes.push_back(y1);
            filterBoxes.push_back(w);
            filterBoxes.push_back(h);

            objProbs.push_back(max_score);
            classId.push_back(max_class_id);
            validCount++;

            // printf("!!! Found Object: score=%f, class=%d\n", max_score, max_class_id); // 发现目标时打印
        }
    }
    return validCount;
}

static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static void compute_dfl(float* tensor, int dfl_len, float* box){
    for (int b=0; b<4; b++){
        float exp_t[dfl_len];
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }
        
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }
}

static int process_u8(uint8_t *box_tensor, int32_t box_zp, float box_scale,
                      uint8_t *score_tensor, int32_t score_zp, float score_scale,
                      uint8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    uint8_t score_thres_u8 = qnt_f32_to_affine_u8(threshold, score_zp, score_scale);
    uint8_t score_sum_thres_u8 = qnt_f32_to_affine_u8(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // Use score sum to quickly filter
            if (score_sum_tensor != nullptr)
            {
                if (score_sum_tensor[offset] < score_sum_thres_u8)
                {
                    continue;
                }
            }

            uint8_t max_score = -score_zp;
            for (int c = 0; c < OBJ_CLASS_NUM; c++)
            {
                if ((score_tensor[offset] > score_thres_u8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score > score_thres_u8)
            {
                offset = i * grid_w + j;
                float box[4];
                float before_dfl[dfl_len * 4];
                for (int k = 0; k < dfl_len * 4; k++)
                {
                    before_dfl[k] = deqnt_affine_u8_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_u8_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes, 
                      std::vector<float> &objProbs, 
                      std::vector<int> &classId, 
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_i8){
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for (int c= 0; c< OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score> score_thres_i8){
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

static int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor, 
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes, 
                        std::vector<float> &objProbs, 
                        std::vector<int> &classId, 
                        float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < threshold){
                    continue;
                }
            }

            float max_score = 0;
            for (int c= 0; c< OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score> threshold){
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = box_tensor[offset];
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

// int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
// {

//     rknn_output *_outputs = (rknn_output *)outputs;
//     std::vector<float> filterBoxes;
//     std::vector<float> objProbs;
//     std::vector<int> classId;
//     int validCount = 0;
//     int stride = 0;
//     int grid_h = 0;
//     int grid_w = 0;
//     int model_in_w = app_ctx->model_width;
//     int model_in_h = app_ctx->model_height;

//     memset(od_results, 0, sizeof(object_detect_result_list));

//     // default 3 branch

//     int dfl_len = app_ctx->output_attrs[0].dims[1] /4;
//     int output_per_branch = app_ctx->io_num.n_output / 3;
//     for (int i = 0; i < 3; i++)
//     {
//         void *score_sum = nullptr;
//         int32_t score_sum_zp = 0;
//         float score_sum_scale = 1.0;
//         if (output_per_branch == 3){
//             score_sum = _outputs[i*output_per_branch + 2].buf;
//             score_sum_zp = app_ctx->output_attrs[i*output_per_branch + 2].zp;
//             score_sum_scale = app_ctx->output_attrs[i*output_per_branch + 2].scale;
//         }
//         int box_idx = i*output_per_branch;
//         int score_idx = i*output_per_branch + 1;


//         grid_h = app_ctx->output_attrs[box_idx].dims[2];
//         grid_w = app_ctx->output_attrs[box_idx].dims[3];

//         stride = model_in_h / grid_h;

//         if (app_ctx->is_quant)
//         {
//             validCount += process_i8((int8_t *)_outputs[box_idx].buf, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
//                                      (int8_t *)_outputs[score_idx].buf, app_ctx->output_attrs[score_idx].zp, app_ctx->output_attrs[score_idx].scale,
//                                      (int8_t *)score_sum, score_sum_zp, score_sum_scale,
//                                      grid_h, grid_w, stride, dfl_len, 
//                                      filterBoxes, objProbs, classId, conf_threshold);
//         }
//         else
//         {
//             validCount += process_fp32((float *)_outputs[box_idx].buf, (float *)_outputs[score_idx].buf, (float *)score_sum,
//                                        grid_h, grid_w, stride, dfl_len, 
//                                        filterBoxes, objProbs, classId, conf_threshold);
//         }
//     }

//     // no object detect
//     if (validCount <= 0)
//     {
//         return 0;
//     }
//     std::vector<int> indexArray;
//     for (int i = 0; i < validCount; ++i)
//     {
//         indexArray.push_back(i);
//     }
//     quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

//     std::set<int> class_set(std::begin(classId), std::end(classId));

//     for (auto c : class_set)
//     {
//         nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
//     }

//     int last_count = 0;
//     od_results->count = 0;

//     /* box valid detect target */
//     for (int i = 0; i < validCount; ++i)
//     {
//         if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
//         {
//             continue;
//         }
//         int n = indexArray[i];

//         float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
//         float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
//         float x2 = x1 + filterBoxes[n * 4 + 2];
//         float y2 = y1 + filterBoxes[n * 4 + 3];
//         int id = classId[n];
//         float obj_conf = objProbs[i];

//         od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
//         od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
//         od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
//         od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
//         od_results->results[last_count].prop = obj_conf;
//         od_results->results[last_count].cls_id = id;
//         last_count++;
//     }
//     od_results->count = last_count;
//     return 0;
// }

int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
{
    rknn_output *_outputs = (rknn_output *)outputs;
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;

    memset(od_results, 0, sizeof(object_detect_result_list));

    // 【核心逻辑修改】
    // 只有一个输出 output[0]
    // 它的维度通常是： dims[0]=1, dims[1]=16 (4+12), dims[2]=8400 (在NCHW下可能不同)
    // 我们需要通过 output_attrs 来判断数据的排列方式 (Permutation)
    
    rknn_tensor_attr* attr = &app_ctx->output_attrs[0];
    int dims[4] = {0};
    // 拷贝 dim 信息，因为 rknn 的 dim 顺序取决于 fmt
    for(int i=0; i<attr->n_dims; i++) dims[i] = attr->dims[i];
    
    int anchor_num = 8400; // 默认值
    int attr_num = 4 + OBJ_CLASS_NUM; // 16
    
    // 简单的维度探测逻辑
    // 找出哪个维度是 8400 (anchor数)，哪个是 16 (属性数)
    int dim_anchor_idx = -1;
    int dim_attr_idx = -1;

    for(int i=0; i<attr->n_dims; i++) {
        if (dims[i] == attr_num) dim_attr_idx = i;
        if (dims[i] > attr_num && dims[i] % 100 == 0) { // 简单假设 anchor 数量较大
             anchor_num = dims[i];
             dim_anchor_idx = i;
        }
    }
    
    if (dim_attr_idx == -1 || dim_anchor_idx == -1) {
        printf("Error: Output tensor shape mismatch! Expect %d classes.\n", OBJ_CLASS_NUM);
        // Fallback: 假设 Ultralytics 默认导出 [1, 16, 8400]
        anchor_num = 8400;
        attr_num = 16;
    }

    // 计算 stride
    // 如果数据是 [1, 16, 8400] (Channel first):
    // Data layout: ch0_anchor0, ch0_anchor1... ch1_anchor0...
    // 访问同一个 anchor 的不同属性: 需跳过 8400 个 float
    // 访问下一个 anchor: 指针 +1
    
    int stride_anchor = 1; 
    int stride_attr = anchor_num; 

    // 如果数据被转置成了 [1, 8400, 16] (Channel last):
    // Data layout: anchor0_ch0, anchor0_ch1...
    // 访问同一个 anchor 的不同属性: 指针 +1
    // 访问下一个 anchor: 需跳过 16 个 float
    if (dim_anchor_idx < dim_attr_idx) { // 例如 [1, 8400, 16]
        stride_anchor = attr_num;
        stride_attr = 1;
    }

    if (app_ctx->is_quant) {
        validCount = process_one_output((int8_t*)_outputs[0].buf, attr->zp, attr->scale,
                                        anchor_num, attr_num, OBJ_CLASS_NUM,
                                        stride_anchor, stride_attr,
                                        filterBoxes, objProbs, classId, conf_threshold);
    } else {
        // 如果 output 是 float 类型
        validCount = process_one_output_fp32((float*)_outputs[0].buf,
                                             anchor_num, attr_num, OBJ_CLASS_NUM,
                                             stride_anchor, stride_attr,
                                             filterBoxes, objProbs, classId, conf_threshold);
    }

    // ---------------- 下面的 NMS 和坐标映射回原图逻辑与之前完全一致 ----------------

    if (validCount <= 0) return 0;

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) indexArray.push_back(i);

    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    od_results->count = 0;

    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) continue;
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.left = (int)(clamp(x1, 0, app_ctx->model_width) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, app_ctx->model_height) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, app_ctx->model_width) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, app_ctx->model_height) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}


int init_post_process()
{
    int ret = 0;
    ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0)
    {
        printf("Load %s failed!\n", LABEL_NALE_TXT_PATH);
        return -1;
    }
    return 0;
}

char *coco_cls_to_name(int cls_id)
{

    if (cls_id >= OBJ_CLASS_NUM)
    {
        return "null";
    }

    if (labels[cls_id])
    {
        return labels[cls_id];
    }

    return "null";
}

void deinit_post_process()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++)
    {
        if (labels[i] != nullptr)
        {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}
