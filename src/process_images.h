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
#include <sys/stat.h>


#ifdef __cplusplus
extern "C" {
#endif

void strlower(char * str);

void process_images_in_dir(char *datacfg, char *cfgfile, char *weightfile, const char * path_to_dir, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void process_image(char *datacfg, char *cfgfile, char *weightfile, const char * path_to_image, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

#endif

#ifdef __cplusplus
}
#endif