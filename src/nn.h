#ifndef NN_H
#define NN_H

typedef enum NN_error_code {
	NN_ERROR_LAYER_INVALID = 20000,
	NN_ERROR_LAYER_NEXT_INVALID,
	NN_ERROR_SAMPLE_INVALID_SIZE,
	NN_ERROR_TF_INVALID,
	NN_ERROR_FILENAME_INVALID
} NN_error_code;

typedef double NN_node;

typedef struct NN_layer {
	int                   nodes_no;
	       NN_node       *stimulus;
	struct NN_layer      *prev;
	struct NN_layer      *next;
} NN_layer;

typedef enum NN_direction {
	NN_FORWARD,
	NN_BACKWARD
} NN_direction;

typedef struct NN_connection {
	struct NN_layer *from;
	struct NN_layer *to;
	double **weights;
} NN_connection;

typedef enum NN_transfer_function {
	NN_TF_LINEAR,
	NN_TF_SIGMOID
} NN_transfer_function;

typedef struct NN_network {
	int                    layers_no;
	double                 lrp; /* learning rate parameter */
	struct NN_layer      **layer;
	struct NN_connection **connection;
	       NN_direction   *direction;
} NN_network;

typedef struct NN_exemplar {
	int inodes_no;
	int onodes_no;
	double *input;
	double *output;
} NN_exemplar;

typedef struct NN_sample {
	int                  size;
	struct NN_exemplar **exemplar;
} NN_sample;

NN_layer      *nn_layer_new                (int nodes_no);
int            nn_layer_zero_stimulus      (NN_layer *layer);
int            nn_layer_stimulate          (NN_layer *layer, NN_transfer_function activation, double *stimulus);
int            nn_layer_propagate          (NN_layer *layer, NN_connection *connection, NN_direction direction, NN_transfer_function output_function);
int            nn_layer_print              (NN_layer *layer);
int            nn_layer_free               (NN_layer *layer);

NN_connection *nn_connection_new           (NN_layer *from, NN_layer *to);
int            nn_connection_randomize     (NN_connection *connection);
int            nn_connection_print         (NN_connection *connection);
int            nn_connection_print_to_file (char *filename, NN_connection *connection);

int            nn_connection_free          (NN_connection *connection);

double         nn_tf_linear                (double in);
double         nn_tf_sigmoid               (double in, double theta, double tau);

NN_network    *nn_network_new              (int layers_no, double lrp);
int            nn_network_train            (NN_network *network, NN_sample *sample, double *cost_function);
int            nn_network_recognize        (NN_network *network, NN_sample *sample);
int            nn_network_free             (NN_network *network);

NN_exemplar   *nn_exemplar_new             (int inodes_no, int onodes_no);
int            nn_exemplar_free            (NN_exemplar *exemplar);

NN_sample     *nn_sample_new               (int size);
int            nn_sample_renorm            (NN_sample *sample, double vmin, double vmax, char *dumpfile);
int            nn_sample_renorm_from_file  (NN_sample *sample, double vmin, double vmax, char *dumpfile);
int            nn_sample_free              (NN_sample *sample);

#endif
