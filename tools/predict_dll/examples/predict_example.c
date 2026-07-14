#include <stdio.h>
#include "axon_predict.h"

int main(void) {
    const char *request =
        "{"
        "\"file\":\"E:/Project/python/Axon_v2.6Exp/sample.exe\","
        "\"checkpoint\":\"E:/Project/python/Axon_v2.6Exp/models/group_isolated_rare_weighted_ft_rebuilt_cache/best_model.pt\","
        "\"device\":\"cpu\","
        "\"scan_nested\":false"
        "}";

    char *response = axon_predict_json(request);
    if (response == NULL) {
        fprintf(stderr, "axon_predict_json returned NULL\n");
        return 1;
    }

    printf("%s\n", response);
    axon_string_free(response);
    return 0;
}
