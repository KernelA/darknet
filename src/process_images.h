#ifndef __PROCESS_IMAGES_H

#define __PROCESS_IMAGES_H

#include <stdlib.h>
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "option_list.h"
#include <libgen.h>
#include <dirent.h>
#include <locale.h>
#include <sys/types.h>


#ifdef __cplusplus
extern "C" {
#endif

void strlower(char * str);

void process_images(char *datacfg, char *cfgfile, char *weightfile, const char * path_to_dir, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

#endif

#ifdef __cplusplus
}
#endif