#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mpi.h>

#define VERBOSE  0
#define VERBOSE2 0
#define VERBOSE3 0
#define VERBOSE4 0
#define VERBOSE5 0

#define THETA 0.75
#define TAU   0.35

int MODE;

char *DATA_DIR;
char *DATA_FORMAT;

int IUNITS;
int HUNITS;
int OUNITS;

double LRP;

int ITERS;
int ITER_OFFSET;

int SAMPLE_SIZE;

double WSEED;

int ADD_JITTER;
double JITTER_SIGMA;
double JITTER_SEED;

int NORMALIZE_INPUT;
int RIGOROUS_CF;

int DUMPSKIP;

int DISABLE_REMAPPING;
int PRINT_EXEMPLARS;

enum {
	TRAINING,
	CONTINUED,
	UNDEFINED
};

void print_synopsis_and_exit ()
{
	printf ("\nUsage: mpirun [MPI_flags] ebai.mpi -r|-t|-c [--other-switches]\n\n");
	printf ("  -h (or -?)             ..  help summary (this screen)\n");
	printf ("  -t                     ..  training mode\n");
	printf ("  -c                     ..  continued training mode\n");
	printf ("  -i iters               ..  number of iterations (t/c modes only)\n");
	printf ("  -s exemplars           ..  number of exemplars (t/c modes) or inputs (r mode)\n");
	printf ("  -n i:h:o               ..  network topology:\n");
	printf ("                             i is the number of input nodes\n");
	printf ("                             h is the number of hidden nodes\n");
	printf ("                             o is the number of output nodes\n");
	printf ("  --lrp value            ..  learning rate parameter value (eta)\n");
	printf ("  --data-dir dirname     ..  sample data directory (for all r/t/c modes)\n");
	printf ("  --data-format format   ..  C-style format for the filename (must have %%d)\n");
	printf ("  --i2h file             ..  use 'file' for input->hidden weights\n");
	printf ("  --h2o file             ..  use 'file' for hidden->output weights\n");
	printf ("  --param-bounds file    ..  use 'file' for parameter bounds (else auto)\n");
	printf ("  --jitter sigma         ..  add synthetic jitter to sample data\n");
	printf ("  --jitter-seed seed     ..  seed to use for randomizing jitter\n");
	printf ("  --weights-seed seed    ..  seed to use for randomizing weights\n");
	printf ("  --iter-offset N        ..  iteration offset for continued training\n");
	printf ("  --dump-i2h prefix      ..  dump I2H weights for each iteration\n");
	printf ("  --dump-h2o prefix      ..  dump H2O weights for each iteration\n");
	printf ("  --dump-skip N          ..  dump weights every N-th iteration\n");
	printf ("  --normalize-input      ..  map the input to the [0,1] interval\n");
	printf ("  --disable-remapping    ..  disable remapping of outputs to [0.1,0.9]\n");
	printf ("  --rigorous-cf          ..  rigorous cost function value computation\n");
	printf ("  --print-exemplars      ..  print exemplars on screen (i.e. debug)\n");
	printf ("\n");
	exit (0);
}

int read_in_exemplars (int num, double *exemplars)
{
	int s, p;
	char fn[255];
	char file_format[255];
	FILE *file;

	sprintf (file_format, "%s/%s", DATA_DIR, DATA_FORMAT);

	for (s = 1; s <= num; s++) {
		sprintf (fn, file_format, s);
		file = fopen (fn, "r");
		if (!file) {
			printf ("File %s does not exist.\n", fn);
			exit (0);
		}

		for (p = 0; p < IUNITS+OUNITS; p++)
		 	fscanf (file, "%lf\n", &(exemplars[(s-1)*(IUNITS+OUNITS)+p]));
		fclose (file);
	}

	return 0;
}

int map_output (double *exemplars, double vmin, double vmax, char *dumpfile)
{
	int p, k;
	double *pmin, *pmax;
	FILE *dump = fopen (dumpfile, "r");

	if (!dump) {
		printf ("parameter bounds file cannot be opened, aborting.\n");
		exit (0);
	}

	pmin = malloc (OUNITS * sizeof (*pmin));
	pmax = malloc (OUNITS * sizeof (*pmax));

	for (k = 0; k < OUNITS; k++)
		fscanf (dump, "%lf\t%lf\n", &(pmin[k]), &(pmax[k]));
	fclose (dump);

	for (p = 0; p < SAMPLE_SIZE; p++)
		for (k = 0; k < OUNITS; k++) {
#if VERBOSE5
			printf ("remapping: %lf -> ", exemplars[IUNITS+(IUNITS+OUNITS)*p+k]);
#endif
			exemplars[IUNITS+(IUNITS+OUNITS)*p+k] = vmin + (vmax - vmin) / (pmax[k]-pmin[k]) * (exemplars[IUNITS+(IUNITS+OUNITS)*p+k]-pmin[k]);
#if VERBOSE5
			printf ("%lf\n", exemplars[IUNITS+(IUNITS+OUNITS)*p+k]);
#endif
		}

	free (pmin);
	free (pmax);

	return 0;
}

int propagate (double *ounits, double *onet, double *hunits, double *hnet, double *i2h, double *h2o, double *exemplars)
{
	int i, j, k;

	for (j = 0; j < HUNITS; j++) {
		hnet[j] = 0.0;
		for (i = 0; i < IUNITS; i++) {
			hnet[j] += i2h[j*IUNITS+i] * exemplars[i];
		}
		hunits[j] = 1./(1.+exp(-(hnet[j]-THETA)/TAU));
#if VERBOSE4
		printf ("%8.3lf", hunits[j]);
#endif
	}
#if VERBOSE4
	printf ("\n");
#endif

	for (k = 0; k < OUNITS; k++) {
		onet[k] = 0.0;
		for (j = 0; j < HUNITS; j++) {
			onet[k] += h2o[k*HUNITS+j] * hunits[j];
		}
		ounits[k] = 1./(1.+exp(-(onet[k]-THETA)/TAU));
#if VERBOSE3
		printf ("%8.3lf", ounits[k]);
#endif
	}
#if VERBOSE3
	printf ("\n");
#endif

	return 0;
}

int main (int argc, char **argv)
{
	int iter, i, j, k, p, n;
	int proc, procno;
	double *exemplars;

	int chunk_size;
	double *chunk;

	char readout_str[255];
	char i2h_filename[255], h2o_filename[255];

	char *param_output_file   = NULL;
	char *i2h_dump_prefix     = NULL;
	char *h2o_dump_prefix     = NULL;

	FILE *i2h_stream, *h2o_stream;

	MPI_Status status;

	/* Initialize MPI stuff: */
	MPI_Init (&argc, &argv);

	/* Get the number of processors and the active processor: */
	MPI_Comm_size (MPI_COMM_WORLD, &procno);
	MPI_Comm_rank (MPI_COMM_WORLD, &proc);

	if (argc < 2 || procno < 2)
		print_synopsis_and_exit ();

	/* Set the defaults: */
	DATA_DIR = strdup ("data");
	DATA_FORMAT = strdup ("dlc%d.data");

	MODE = UNDEFINED;

	IUNITS = 201;
	HUNITS = 40;
	OUNITS = 5;

	LRP = 0.15;

	ITER_OFFSET = 0;
	ITERS = 300000;

	SAMPLE_SIZE = 10000;

	ADD_JITTER = 0;
	JITTER_SIGMA = 0.0;
	JITTER_SEED = time (0);

	WSEED = time (0);

	NORMALIZE_INPUT = 0;
	RIGOROUS_CF = 0;

	DUMPSKIP = 1;

	DISABLE_REMAPPING = 0;
	PRINT_EXEMPLARS = 0;

	sprintf (i2h_filename, "i2h.weights");
	sprintf (h2o_filename, "h2o.weights");

	/* Parse the command line: */
	for (i = 1; i < argc; i++) {
		if (strcmp (argv[i], "-h") == 0 || strcmp (argv[i], "-?") == 0)
			print_synopsis_and_exit ();
		else if (strcmp (argv[i], "-t") == 0)
			MODE = TRAINING;
		else if (strcmp (argv[i], "-c") == 0)
			MODE = CONTINUED;
		else if (strcmp (argv[i], "-i") == 0) {
			i++;
			sscanf (argv[i], "%d", &ITERS);
		}
		else if (strcmp (argv[i], "-s") == 0) {
			i++;
			sscanf (argv[i], "%d", &SAMPLE_SIZE);
		}
		else if (strcmp (argv[i], "-n") == 0) {
			i++;
			sscanf (argv[i], "%d:%d:%d", &IUNITS, &HUNITS, &OUNITS);
		}
		else if (strcmp (argv[i], "--lrp") == 0) {
			i++;
			sscanf (argv[i], "%lf", &LRP);
		}
		else if (strcmp (argv[i], "--data-dir") == 0) {
			i++;
			sscanf (argv[i], "%s", &(readout_str[0]));
			free (DATA_DIR);
			DATA_DIR = strdup (readout_str);
		}
		else if (strcmp (argv[i], "--data-format") == 0) {
			i++;
			sscanf (argv[i], "%s", &(readout_str[0]));
			free (DATA_FORMAT);
			DATA_FORMAT = strdup (readout_str);
		}
		else if (strcmp (argv[i], "--i2h") == 0) {
			i++;
			sscanf (argv[i], "%s", &(i2h_filename[0]));
		}
		else if (strcmp (argv[i], "--h2o") == 0) {
			i++;
			sscanf (argv[i], "%s", &(h2o_filename[0]));
		}
		else if (strcmp (argv[i], "--param-bounds") == 0) {
			i++;
			sscanf (argv[i], "%s", &(readout_str[0]));
			param_output_file = strdup (readout_str);
		}
		else if (strcmp (argv[i], "--jitter") == 0) {
			ADD_JITTER = 1;
			i++;
			sscanf (argv[i], "%lf", &JITTER_SIGMA);
		}
		else if (strcmp (argv[i], "--jitter-seed") == 0) {
			i++;
			sscanf (argv[i], "%lf", &JITTER_SEED);
		}
		else if (strcmp (argv[i], "--weights-seed") == 0) {
			i++;
			sscanf (argv[i], "%lf", &WSEED);
		}
		else if (strcmp (argv[i], "--dump-i2h") == 0) {
			i++;
			sscanf (argv[i], "%s", &(readout_str[0]));
			i2h_dump_prefix = strdup (readout_str);
		}
		else if (strcmp (argv[i], "--dump-h2o") == 0) {
			i++;
			sscanf (argv[i], "%s", &(readout_str[0]));
			h2o_dump_prefix = strdup (readout_str);
		}
		else if (strcmp (argv[i], "--dump-skip") == 0) {
			i++;
			sscanf (argv[i], "%d", &DUMPSKIP);
		}
		else if (strcmp (argv[i], "--iter-offset") == 0) {
			i++;
			sscanf (argv[i], "%d", &ITER_OFFSET);
		}
		else if (strcmp (argv[i], "--normalize-input") == 0) {
			NORMALIZE_INPUT = 1;
		}
		else if (strcmp (argv[i], "--rigorous-cf") == 0) {
			RIGOROUS_CF = 1;
		}
		else if (strcmp (argv[i], "--disable-remapping") == 0) {
			DISABLE_REMAPPING = 1;
		}
		else if (strcmp (argv[i], "--print-exemplars") == 0) {
			PRINT_EXEMPLARS = 1;
		}
		else {
			printf ("\nSwitch %s not recognized, aborting.\n\n", argv[i]);
			exit (0);
		}
	}

	if (proc == 0) {
		double **i2h, **h2o, *cf;
		double sum;

		printf ("# Issued command:\n");
		printf ("#   ");
		for (i = 0; i < argc; i++)
			printf ("%s ", argv[i]);
		printf ("\n# \n");

		printf ("# building a neural network: %d input nodes, %d hidden nodes, %d output nodes\n", IUNITS, HUNITS, OUNITS);
		printf ("# learning rate parameter (eta): %lf\n", LRP);
		printf ("# data directory: %s\n", DATA_DIR);
		printf ("# data filename format: %s\n", DATA_FORMAT);

		/* Allocate memory for exemplars: */
		exemplars = malloc (SAMPLE_SIZE * (IUNITS+OUNITS) * sizeof (*exemplars));

		/* Read in exemplars: */
		read_in_exemplars (SAMPLE_SIZE, exemplars);

		printf ("# number of exemplars read: %d\n", SAMPLE_SIZE);

		/* Remap exemplars to [0.1,0.9]: */
		map_output (exemplars, 0.1, 0.9, param_output_file);

		/* Allocate memory for weights: */
		i2h = malloc (procno * sizeof (*i2h));
		h2o = malloc (procno * sizeof (*h2o));
		 cf = malloc (procno * sizeof ( *cf));
		for (n = 0; n < procno; n++) {
			i2h[n] = malloc (IUNITS*HUNITS * sizeof(**i2h));
			h2o[n] = malloc (HUNITS*OUNITS * sizeof(**h2o));
		}

		if (MODE == TRAINING) {
			/* Randomize weights: */
			srand (WSEED);
			for (i = 0; i < IUNITS*HUNITS; i++)
				i2h[0][i] = -0.5 + (double) rand()/RAND_MAX;
			for (i = 0; i < HUNITS*OUNITS; i++)
				h2o[0][i] = -0.5 + (double) rand()/RAND_MAX;
		}
		if (MODE == CONTINUED) {
			i2h_stream = fopen (i2h_filename, "r");
			for (j = 0; j < HUNITS; j++) {
				for (i = 0; i < IUNITS; i++)
					fscanf (i2h_stream, "%lf", &(i2h[0][j*IUNITS+i]));
				fscanf (i2h_stream, "\n");
			}
			fclose (i2h_stream);

			h2o_stream = fopen (h2o_filename, "r");
			for (k = 0; k < OUNITS; k++) {
				for (j = 0; j < HUNITS; j++)
					fscanf (h2o_stream, "%lf", &(h2o[0][k*HUNITS+j]));
				fscanf (h2o_stream, "\n");
			}
			fclose (h2o_stream);
		}

		/* Partition the input for MPI propagation; since exemplars remain the
		 * same throughout backpropagation, we do this outside the iterative
		 * loop.
		 */

		chunk_size = SAMPLE_SIZE / (procno-1);
		printf ("# number of procesors: %d\n", procno);
		printf ("# number of exemplars per processor: %d\n", chunk_size);
		for (n = 1; n < procno; n++) {
			chunk = exemplars + (n-1)*chunk_size*(IUNITS+OUNITS);
			MPI_Send (chunk, chunk_size*(IUNITS+OUNITS), MPI_DOUBLE, n, 15+n, MPI_COMM_WORLD);
		}

		/* This is where we start our iterative back-propagation. */
		for (iter = 0; iter < ITERS; iter++) {
			/* Broadcast the initial propagation matrices to all nodes: */
			MPI_Bcast (i2h[0], IUNITS*HUNITS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast (h2o[0], HUNITS*OUNITS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			/* Collect the results: */
			for (n = 1; n < procno; n++) {
				MPI_Recv (i2h[n], IUNITS*HUNITS, MPI_DOUBLE, n, n,     MPI_COMM_WORLD, &status);
				MPI_Recv (h2o[n], HUNITS*OUNITS, MPI_DOUBLE, n, 100+n, MPI_COMM_WORLD, &status);
				MPI_Recv (&cf[n],       1      , MPI_DOUBLE, n, 200+n, MPI_COMM_WORLD, &status);
			}

			/* Merge the results: */
			for (j = 0; j < HUNITS; j++)
				for (i = 0; i < IUNITS; i++) {
					sum = 0.0;
					for (n = 1; n < procno; n++)
						sum += i2h[n][j*IUNITS+i];
					i2h[0][j*IUNITS+i] = sum/(procno-1);
				}

			for (k = 0; k < OUNITS; k++)
				for (j = 0; j < HUNITS; j++) {
					sum = 0.0;
					for (n = 1; n < procno; n++)
						sum += h2o[n][k*HUNITS+j];
					h2o[0][k*HUNITS+j] = sum/(procno-1);
				}

			if (RIGOROUS_CF) {
				double ounits[5], onet[5], hunits[40], hnet[40], dif;
				cf[0] = 0.0;
				for (p = 0; p < SAMPLE_SIZE; p++) {
					propagate (ounits, onet, hunits, hnet, i2h[0], h2o[0], &(exemplars[p*(IUNITS+OUNITS)]));
					for (k = 0; k < OUNITS; k++) {
						dif = exemplars[p*(IUNITS+OUNITS)+IUNITS+k]-ounits[k];
						cf[0] += dif*dif;
					}
				}
			}
			else {
				cf[0] = 0.0;
				for (n = 1; n < procno; n++)
					cf[0] += cf[n];
			}

			printf ("%d\t%e\n", ITER_OFFSET+iter, cf[0]);
		}

		i2h_stream = fopen (i2h_filename, "w");
		for (j = 0; j < HUNITS; j++) {
			for (i = 0; i < IUNITS; i++)
				fprintf (i2h_stream, "%12.8lf", i2h[0][j*IUNITS+i]);
			fprintf (i2h_stream, "\n");
		}
		fclose (i2h_stream);

		h2o_stream = fopen (h2o_filename, "w");
		for (k = 0; k < OUNITS; k++) {
			for (j = 0; j < HUNITS; j++)
				fprintf (h2o_stream, "%12.8lf", h2o[0][k*HUNITS+j]);
			fprintf (h2o_stream, "\n");
		}
		fclose (h2o_stream);
	}
	else {
		/* This part belongs to work nodes. */
		double *i2h, *h2o;
		double *ounits, *onet, *odelta;
		double *hunits, *hnet, *hdelta;
		double dif, cf;

		/* Allocate memory for initial weights: */
		i2h = malloc (IUNITS * HUNITS * sizeof (*i2h));
		h2o = malloc (HUNITS * OUNITS * sizeof (*h2o));

#if VERBOSE2
		printf ("Processor %d says:\n", proc);
#endif

		/* Allocate memory for the chunk of exemplars: */
		chunk_size = SAMPLE_SIZE / (procno-1);
		exemplars = malloc (chunk_size * (IUNITS+OUNITS) * sizeof(*exemplars));

		/* Get the chunk from the root: */
		MPI_Recv (exemplars, chunk_size*(IUNITS+OUNITS), MPI_DOUBLE, 0, 15+proc, MPI_COMM_WORLD, &status);

#if VERBOSE2
		printf ("  I got %d/%d exemplars in my chunk:\n", chunk_size, EXEMPLARS);
		for (p = 0; p < chunk_size; p++) {
			printf ("  %d: ", p);
			for (i = 0; i < IUNITS; i++)
				printf ("%8.3lf", exemplars[p*(IUNITS+OUNITS)+i]);
			printf ("   ;");
			for (k = IUNITS; k < IUNITS+OUNITS; k++)
				printf ("%8.3lf", exemplars[p*(IUNITS+OUNITS)+k]);
			printf ("\n");
		}
#endif

		/* Allocate memory for net values, layer values, and deltas: */
		hnet   = malloc (HUNITS * sizeof(*hnet));
		hunits = malloc (HUNITS * sizeof(*hunits));
		onet   = malloc (OUNITS * sizeof(*onet));
		ounits = malloc (OUNITS * sizeof(*ounits));
		odelta = malloc (OUNITS * sizeof(*odelta));
		hdelta = malloc (HUNITS * sizeof(*hdelta));

		/* Here we start with the iterations. */
		for (iter = 0; iter < ITERS; iter++) {
			/* Get weights from broadcast: */

			MPI_Bcast (i2h, IUNITS*HUNITS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast (h2o, HUNITS*OUNITS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			/* Do the backpropagation. */
			cf = 0.0;
			for (p = 0; p < chunk_size; p++) {
				/* First task for the compute nodes is forward propagation. */
				propagate (ounits, onet, hunits, hnet, i2h, h2o, &(exemplars[p*(IUNITS+OUNITS)]));
#if VERBOSE2
				printf ("  delO(%d): ", p);
#endif
				for (k = 0; k < OUNITS; k++) {
					dif = exemplars[p*(IUNITS+OUNITS)+IUNITS+k]-ounits[k];
					cf  += dif*dif;
					odelta[k] = dif*ounits[k]*(1.0-ounits[k]);
#if VERBOSE2
					printf ("%8.3lf", odelta[k]);
#endif
				}
#if VERBOSE2
				printf ("\n");
				printf ("  delH(%d): ", p);
#endif
				for (j = 0; j < HUNITS; j++) {
					hdelta[j] = 0.0;
					for (k = 0; k < OUNITS; k++)
						hdelta[j] += h2o[k*HUNITS+j] * odelta[k];
					hdelta[j] *= hunits[j] * (1.0-hunits[j]);
#if VERBOSE2
					printf ("%8.3lf", hdelta[j]);
#endif
				}
#if VERBOSE2
				printf ("\n");
#endif
				for (k = 0; k < OUNITS; k++)
					for (j = 0; j < HUNITS; j++)
						h2o[k*HUNITS+j] += LRP * odelta[k] * hunits[j];
				for (j = 0; j < HUNITS; j++)
					for (i = 0; i < IUNITS; i++)
						i2h[j*IUNITS+i] += LRP * hdelta[j] * exemplars[p*(IUNITS+OUNITS)+i];
#if VERBOSE
				printf ("  New values:\n");
				printf ("  I2H: ");
				for (j = 0; j < HUNITS; j++) {
					for (i = 0; i < IUNITS; i++) {
						printf ("%8.3lf", i2h[j*IUNITS+i]);
					}
					printf ("\n");
					if (j != HUNITS-1) printf ("       ");
				}
				printf ("  H2O: ");
				for (k = 0; k < OUNITS; k++) {
					for (j = 0; j < HUNITS; j++) {
						printf ("%8.3lf", h2o[k*HUNITS+j]);
					}
					printf ("\n");
					if (k != OUNITS-1) printf ("       ");
				}
#endif
			}

			/* Final step: send these results back to root for merging. */
			MPI_Send (i2h, IUNITS*HUNITS, MPI_DOUBLE, 0, proc, MPI_COMM_WORLD);
			MPI_Send (h2o, HUNITS*OUNITS, MPI_DOUBLE, 0, 100+proc, MPI_COMM_WORLD);
			MPI_Send (&cf,       1      , MPI_DOUBLE, 0, 200+proc, MPI_COMM_WORLD);
		} /* End of iterative block */

		free (hnet);
		free (hunits);
		free (onet);
		free (ounits);
		free (odelta);
		free (hdelta);

		free (i2h);
		free (h2o);
		free (exemplars);
	} /* End of compute node block */

	MPI_Finalize ();

	return 0;
}
