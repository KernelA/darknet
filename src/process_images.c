#include "process_images.h"

void strlower(char *str)
{
    char *symbol = str;

    while (*symbol != '\0')
    {
        *symbol = tolower(*symbol);
        symbol++;
    }
}

void process_images_in_dir(char *datacfg, char *cfgfile, char *weightfile, const char *path_to_dir, float thresh,
                    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    if (path_to_dir == NULL)
    {
        printf("\n Directory is not a valid\n");
        return;
    }

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = NULL;
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1

    if (weightfile)
    {
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size)
    {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
               name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size)
            getchar();
    }
    srand(2222222);

    char *json_buf = NULL;
    int json_image_id = 0;

    FILE *json_file = NULL;
    if (outfile)
    {
        json_file = fopen(outfile, "wb");
        if (!json_file)
        {
            error("fopen failed");
        }
        char *tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    int j;
    float nms = .45; // 0.4F

    const int buffer_size = 4096;
    char new_name_buffer[buffer_size];

    printf("Try to open %s\n", path_to_dir);

    DIR *dir = opendir(path_to_dir);

    if (dir == NULL)
    {
        printf("Cannot open directory %s", path_to_dir);
        return;
    }

    struct dirent *entry = readdir(dir);

    if (entry == NULL)
    {
        printf("Cannot open any entry in directory\n");
    }

    while (entry)
    {
        char *file_name = basename(entry->d_name);
        char *last_dot = strrchr(file_name, '.');

        if (last_dot)
        {
            char *extension = (char *)malloc(sizeof(char) * strlen(last_dot));

            if (extension == NULL)
            {
                printf("Cannot allocate memory for extension.");
                continue;
            }

            strcpy(extension, last_dot);
            strlower(extension);

            if (strcmp(extension, ".jpg") == 0)
            {
                free(extension);
                extension = NULL;

                sprintf(new_name_buffer, "%s/%s", path_to_dir, file_name);
                printf("Read: %s\n", new_name_buffer);

                image im = load_image(new_name_buffer, 0, 0, net.c);
                image sized;

                if (letter_box)
                    sized = letterbox_image(im, net.w, net.h);
                else
                    sized = resize_image(im, net.w, net.h);
                layer l = net.layers[net.n - 1];

                float *X = sized.data;

                //time= what_time_is_it_now();
                double time = get_time_point();
                network_predict(net, X);
                //network_predict_image(&net, im); letterbox = 1;
                printf("%s: Predicted in %lf milli-seconds.\n", file_name, ((double)get_time_point() - time) / 1000);
                //printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));

                int nboxes = 0;
                detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
                if (nms)
                {
                    if (l.nms_kind == DEFAULT_NMS)
                        do_nms_sort(dets, nboxes, l.classes, nms);
                    else
                        diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
                }

                draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);

                if (!dont_show)
                {
                    show_image(im, "predictions");
                }

                if (json_file)
                {
                    if (json_buf)
                    {
                        char *tmp = ", \n";
                        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
                    }
                    ++json_image_id;
                    json_buf = detection_to_json(dets, nboxes, l.classes, names, json_image_id, file_name);

                    fwrite(json_buf, sizeof(char), strlen(json_buf), json_file);
                    free(json_buf);
                }

                // pseudo labeling concept - fast.ai
                if (save_labels)
                {
                    char labelpath[4096];
                    replace_image_to_label(entry->d_name, labelpath);

                    FILE *fw = fopen(labelpath, "wb");
                    int i;
                    for (i = 0; i < nboxes; ++i)
                    {
                        char buff[1024];
                        int class_id = -1;
                        float prob = 0;
                        for (j = 0; j < l.classes; ++j)
                        {
                            if (dets[i].prob[j] > thresh && dets[i].prob[j] > prob)
                            {
                                prob = dets[i].prob[j];
                                class_id = j;
                            }
                        }
                        if (class_id >= 0)
                        {
                            sprintf(buff, "%d %2.4f %2.4f %2.4f %2.4f\n", class_id, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
                            fwrite(buff, sizeof(char), strlen(buff), fw);
                        }
                    }
                    fclose(fw);
                }

                free_detections(dets, nboxes);
                free_image(im);
                free_image(sized);

                if (!dont_show)
                {
                    wait_until_press_key_cv();
                    destroy_all_windows_cv();
                }
            }
        }

        entry = readdir(dir);
    }

    closedir(dir);

    if (json_file)
    {
        char *tmp = "\n]";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
        fclose(json_file);
        printf("Save all predictions to %s\n", outfile);
    }

    // free memory
    free_ptrs((void **)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    if (alphabet != NULL)
    {
        int i;
        const int nsize = 8;
        for (j = 0; j < nsize; ++j)
        {
            for (i = 32; i < 127; ++i)
            {
                free_image(alphabet[j][i]);
            }
            free(alphabet[j]);
        }
        free(alphabet);
    }

    free_network(net);
}

void process_image(char *datacfg, char *cfgfile, char *weightfile, const char *path_to_image, float thresh,
                   float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    if (path_to_image == NULL)
    {
        printf("\n Input file is not a valid\n");
        return;
    }

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = NULL;
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1

    if (weightfile)
    {
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size)
    {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
               name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size)
            getchar();
    }
    srand(2222222);

    char *json_buf = NULL;
    int json_image_id = 0;

    FILE *json_file = NULL;
    if (outfile)
    {
        json_file = fopen(outfile, "wb");
        if (!json_file)
        {
            error("fopen failed");
        }
        char *tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    int j;
    float nms = .45; // 0.4F

    char * copy_image = (char*)malloc(sizeof(char) * strlen(path_to_image));

    if(copy_image == NULL)
    {
        printf("Cannot allocate memory for new name.");
        goto free_resources;
    }

    strcpy(copy_image, path_to_image);

    char *file_name = basename(copy_image);
    free(copy_image);
    copy_image = NULL;
    char *last_dot = strrchr(file_name, '.');

    if (last_dot)
    {
        char *extension = (char *)malloc(sizeof(char) * strlen(last_dot));

        if (extension == NULL)
        {
            printf("Cannot allocate memory for extension.");
            goto free_resources;
        }

        strcpy(extension, last_dot);
        strlower(extension);

        if (strcmp(extension, ".jpg") == 0)
        {
            free(extension);
            extension = NULL;

            printf("Read: %s\n", path_to_image);

            image im = load_image((char*)path_to_image, 0, 0, net.c);
            image sized;

            if (letter_box)
                sized = letterbox_image(im, net.w, net.h);
            else
                sized = resize_image(im, net.w, net.h);
            layer l = net.layers[net.n - 1];

            float *X = sized.data;

            //time= what_time_is_it_now();
            double time = get_time_point();
            network_predict(net, X);
            //network_predict_image(&net, im); letterbox = 1;
            printf("%s: Predicted in %lf milli-seconds.\n", file_name, ((double)get_time_point() - time) / 1000);
            //printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));

            int nboxes = 0;
            detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
            if (nms)
            {
                if (l.nms_kind == DEFAULT_NMS)
                    do_nms_sort(dets, nboxes, l.classes, nms);
                else
                    diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }

            draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);

            if (!dont_show)
            {
                show_image(im, "predictions");
            }

            if (json_file)
            {
                if (json_buf)
                {
                    char *tmp = ", \n";
                    fwrite(tmp, sizeof(char), strlen(tmp), json_file);
                }
                ++json_image_id;
                json_buf = detection_to_json(dets, nboxes, l.classes, names, json_image_id, file_name);

                fwrite(json_buf, sizeof(char), strlen(json_buf), json_file);
                free(json_buf);
            }

            // pseudo labeling concept - fast.ai
            if (save_labels)
            {
                char labelpath[4096];
                replace_image_to_label(path_to_image, labelpath);

                FILE *fw = fopen(labelpath, "wb");
                int i;
                for (i = 0; i < nboxes; ++i)
                {
                    char buff[1024];
                    int class_id = -1;
                    float prob = 0;
                    for (j = 0; j < l.classes; ++j)
                    {
                        if (dets[i].prob[j] > thresh && dets[i].prob[j] > prob)
                        {
                            prob = dets[i].prob[j];
                            class_id = j;
                        }
                    }
                    if (class_id >= 0)
                    {
                        sprintf(buff, "%d %2.4f %2.4f %2.4f %2.4f\n", class_id, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
                        fwrite(buff, sizeof(char), strlen(buff), fw);
                    }
                }
                fclose(fw);
            }

            free_detections(dets, nboxes);
            free_image(im);
            free_image(sized);

            if (!dont_show)
            {
                wait_until_press_key_cv();
                destroy_all_windows_cv();
            }
        }
    }

free_resources:
    if (json_file)
    {
        char *tmp = "\n]";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
        fclose(json_file);
        printf("Save all predictions to %s\n", outfile);
    }

    // free memory
    free_ptrs((void **)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    if (alphabet != NULL)
    {
        int i;
        const int nsize = 8;
        for (j = 0; j < nsize; ++j)
        {
            for (i = 32; i < 127; ++i)
            {
                free_image(alphabet[j][i]);
            }
            free(alphabet[j]);
        }
        free(alphabet);
    }

    free_network(net);
}