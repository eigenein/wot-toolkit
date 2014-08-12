#include <stdlib.h>
#include <math.h>

#include <Python.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    /* row count */              int row_count;
    /* column count */           int column_count;
    /* value count */            int value_count;
    /* row indexes */            int *rows;
    /* column indexes */         int *columns;
    /* rating values */          double *values;
    /* base predictor */         double base;
    /* row base predictors */    double *row_bases;
    /* column base predictors */ double *column_bases;
    /* feature count*/           int feature_count;
    /* regularization */         double lambda;
    /* learned features */       double *row_features;
    /* learned features */       double *column_features;
    /* Beta features: */
    /* distribution levels */    double distribution_levels[100];
    /* distribution */           int distribution[100];
} Model;

/*
  Model initialization and deallocation.
--------------------------------------------------------------------------------
 */

static PyObject *
model_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Model *self = (Model*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->row_count = self->column_count = self->value_count = 0;
        self->rows = self->columns = NULL;
        self->values = NULL;
        self->base = 0.0;
        self->row_bases = self->column_bases = NULL;
        self->feature_count = 0;
        self->row_features = self->column_features = NULL;
    }
    return (PyObject*)self;
}

int alloc_wrapper(int n, void **p) {
    *p = PyMem_RawMalloc(n);
    if (*p != NULL) {
        memset(*p, 0, n);
        return 1;
    } else {
        PyErr_SetString(PyExc_MemoryError, "not enough memory");
        return 0;
    }
}

static int
model_init(Model *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"row_count", "column_count", "value_count", "feature_count", "_lambda", NULL};
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "iiiid", kwlist, 
        &self->row_count, &self->column_count, &self->value_count, &self->feature_count, &self->lambda)) {
        return -1;
    }
    // Allocate memory.
    if (
        !alloc_wrapper(self->value_count * sizeof(int), (void**)&self->rows) ||
        !alloc_wrapper(self->value_count * sizeof(int), (void**)&self->columns) ||
        !alloc_wrapper(self->value_count * sizeof(double), (void**)&self->values) ||
        !alloc_wrapper(self->row_count * sizeof(double), (void**)&self->row_bases) ||
        !alloc_wrapper(self->column_count * sizeof(double), (void**)&self->column_bases) ||
        !alloc_wrapper(self->row_count * self->feature_count * sizeof(double), (void**)&self->row_features) ||
        !alloc_wrapper(self->column_count * self->feature_count * sizeof(double), (void**)&self->column_features)
    ) {
        return -1;
    }
    return 0;
}

static void
model_dealloc(Model *self) {
    PyMem_RawFree(self->rows);
    PyMem_RawFree(self->columns);
    PyMem_RawFree(self->values);
    PyMem_RawFree(self->row_bases);
    PyMem_RawFree(self->column_bases);
    PyMem_RawFree(self->row_features);
    PyMem_RawFree(self->column_features);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/*
  Model setters.
--------------------------------------------------------------------------------
 */

static PyObject *
model_set_value(Model *self, PyObject *args, PyObject *kwargs) {
    int index;
    long row, column;
    double value;

    static char *kwlist[] = {"index", "row", "column", "value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiid", kwlist, &index, &row, &column, &value)) {
        return NULL;
    }

    self->rows[index] = row;
    self->columns[index] = column;
    self->values[index] = value;

    Py_RETURN_NONE;
}

static PyObject *
model_set_distribution_level(Model *self, PyObject *args, PyObject *kwargs) {
    int i;
    double level;

    static char *kwlist[] = {"i", "level", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "id", kwlist, &i, &level)) {
        return NULL;
    }

    self->distribution_levels[i] = level;

    Py_RETURN_NONE;
}

/*
  Model methods.
--------------------------------------------------------------------------------
 */

double
rand_wrapper(double randomness) {
    return randomness * (1.0 * rand() / RAND_MAX - 0.5);
}

static PyObject *
model_prepare(Model *self, PyObject *args, PyObject *kwargs) {
    double randomness;

    static char *kwlist[] = {"randomness", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &randomness)) {
        return NULL;
    }
    // Randomize base.
    self->base = rand_wrapper(randomness);
    // Randomize row bases.
    for (int i = 0; i < self->row_count; i++) {
        self->row_bases[i] = rand_wrapper(randomness);
    }
    // Randomize column bases.
    for (int i = 0; i < self->column_count; i++) {
        self->column_bases[i] = rand_wrapper(randomness);
    }
    // Randomize row features.
    for (int i = 0; i < self->row_count * self->feature_count; i++) {
        self->row_features[i] = rand_wrapper(randomness);
    }
    // Randomize column features.
    for (int i = 0; i < self->column_count * self->feature_count; i++) {
        self->column_features[i] = rand_wrapper(randomness);
    }

    Py_RETURN_NONE;
}

#define SWAP(array, i, j, temp_type) { const temp_type temp = array[i]; array[i] = array[j]; array[j] = temp; }

static PyObject *
model_shuffle(Model *self, PyObject *args, PyObject *kwargs) {
    int start, stop;

    static char *kwlist[] = {"start", "stop", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &start, &stop)) {
        return NULL;
    }

    for (int i = stop - 1; i != start; i--) {
        int j = rand() % (i + 1);
        SWAP(self->rows, i, j, int);
        SWAP(self->columns, i, j, int);
        SWAP(self->values, i, j, double);
    }

    Py_RETURN_NONE;
}

double features_dot(Model *self, int row, int column) {
    double dot = 0.0;
    
    for (int i = 0; i < self->feature_count; i++) {
        dot += 
            self->row_features[row * self->feature_count + i] *
            self->column_features[column * self->feature_count + i];
    }

    return dot;
}

static PyObject *
model_step(Model *self, PyObject *args, PyObject *kwargs) {
    int start, stop;
    double alpha;
    double rmse = 0.0, min_error = INFINITY, average_error = 0.0, max_error = 0.0;

    static char *kwlist[] = {"start", "stop", "alpha", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iid", kwlist, &start, &stop, &alpha)) {
        return NULL;
    }

    memset(self->distribution, 0, sizeof(self->distribution));

    for (int i = start; i < stop; i++) {
        int row = self->rows[i];
        int column = self->columns[i];
        // Update error.
        double error = self->values[i] - (
            self->base + self->row_bases[row] + self->column_bases[column] + features_dot(self, row, column));
        rmse += error * error;
        // Update base predictors.
        self->base += alpha * error;
        self->row_bases[row] += alpha * (error - self->lambda * self->row_bases[row]);
        self->column_bases[column] += alpha * (error - self->lambda * self->column_bases[column]);
        // Update features.
        for (int j = 0; j < self->feature_count; j++) {
            int row_offset = row * self->feature_count + j;
            int column_offset = column * self->feature_count + j;
            double row_feature = self->row_features[row_offset];
            self->row_features[row_offset] += alpha * (
                error * self->column_features[column_offset] - self->lambda * self->row_features[row_offset]);
            self->column_features[column_offset] += alpha * (
                error * row_feature - self->lambda * self->column_features[column_offset]);
        }
        // Statistics.
        double abs_error = fabs(error);
        min_error = fmin(min_error, abs_error);
        average_error += abs_error;
        max_error = fmax(max_error, abs_error);
        // Distribution.
        for (int j = 0; j < 100; j++) {
            if (abs_error < self->distribution_levels[j]) {
                self->distribution[j] += 1;
            }
        }
    }
    // Return error.
    rmse /= self->value_count;
    average_error /= self->value_count;
    return Py_BuildValue("(dddd)", rmse, min_error, average_error, max_error);
}

/*
  Model getters.
--------------------------------------------------------------------------------
 */

static PyObject *
model_get_row_base(Model *self, PyObject *args, PyObject *kwargs) {
    int row;
    static char *kwlist[] = {"row", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &row)) {
        return NULL;
    }
    return PyFloat_FromDouble(self->row_bases[row]);
}

static PyObject *
model_get_column_base(Model *self, PyObject *args, PyObject *kwargs) {
    int column;
    static char *kwlist[] = {"column", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &column)) {
        return NULL;
    }
    return PyFloat_FromDouble(self->column_bases[column]);
}

static PyObject *
model_get_row_feature(Model *self, PyObject *args, PyObject *kwargs) {
    int row, j;
    static char *kwlist[] = {"row", "j", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &row, &j)) {
        return NULL;
    }
    return PyFloat_FromDouble(self->row_features[row * self->feature_count + j]);
}

static PyObject *
model_get_column_feature(Model *self, PyObject *args, PyObject *kwargs) {
    int column, j;
    static char *kwlist[] = {"column", "j", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &column, &j)) {
        return NULL;
    }
    return PyFloat_FromDouble(self->column_features[column * self->feature_count + j]);
}

static PyObject *
model_get_distribution(Model *self, PyObject *args, PyObject *kwargs) {
    int i;
    static char *kwlist[] = {"i", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) {
        return NULL;
    }
    return PyLong_FromLong(self->distribution[i]);
}

/*
  Predicting.
--------------------------------------------------------------------------------
 */

static PyObject *
model_predict(Model *self, PyObject *args, PyObject *kwargs) {
    int row, column;
    static char *kwlist[] = {"row", "column", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &row, &column)) {
        return NULL;
    }
    return PyFloat_FromDouble(self->base + self->row_bases[row] + self->column_bases[column] + features_dot(self, row, column));
}

/*
  Model definition.
--------------------------------------------------------------------------------
 */

static PyMemberDef model_members[] = {
    {"row_count", T_INT, offsetof(Model, row_count), 0, "Row count."},
    {"column_count", T_INT, offsetof(Model, column_count), 0, "Column count."},
    {"value_count", T_INT, offsetof(Model, value_count), 0, "Value count."},
    {"base", T_DOUBLE, offsetof(Model, base), 0, "Learned base predictor."},
    {NULL}
};

static PyMethodDef model_methods[] = {
    {"set_value", (PyCFunction)model_set_value, METH_VARARGS | METH_KEYWORDS, "Sets value."},
    {"prepare", (PyCFunction)model_prepare, METH_VARARGS | METH_KEYWORDS, "Prepares model for training."},
    {"shuffle", (PyCFunction)model_shuffle, METH_VARARGS | METH_KEYWORDS, "Shuffles values."},
    {"step", (PyCFunction)model_step, METH_VARARGS | METH_KEYWORDS, "Does gradient descent step."},
    {"get_row_base", (PyCFunction)model_get_row_base, METH_VARARGS | METH_KEYWORDS, "Gets learned row base predictor."},
    {"get_column_base", (PyCFunction)model_get_column_base, METH_VARARGS | METH_KEYWORDS, "Gets learned column base predictor."},
    {"get_row_feature", (PyCFunction)model_get_row_feature, METH_VARARGS | METH_KEYWORDS, "Gets learned row feature."},
    {"get_column_feature", (PyCFunction)model_get_column_feature, METH_VARARGS | METH_KEYWORDS, "Gets learned column feature."},
    {"predict", (PyCFunction)model_predict, METH_VARARGS | METH_KEYWORDS, "Predicts rating."},
    {"set_distribution_level", (PyCFunction)model_set_distribution_level, METH_VARARGS | METH_KEYWORDS, "Sets distribution level."},
    {"get_distribution", (PyCFunction)model_get_distribution, METH_VARARGS | METH_KEYWORDS, "Gets distribution item."},
    {NULL}
};

static PyTypeObject ModelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /* tp_name */          "trainer.Model",
    /* tp_basicsize */      sizeof(Model),
    /* tp_itemsize */       0,
    /* tp_dealloc */        (destructor)model_dealloc,
    /* tp_print */          0,
    /* tp_getattr */        0,
    /* tp_setattr */        0,
    /* tp_reserved */       0,
    /* tp_repr */           0,
    /* tp_as_number */      0,
    /* tp_as_sequence */    0,
    /* tp_as_mapping */     0,
    /* tp_hash  */          0,
    /* tp_call */           0,
    /* tp_str */            0,
    /* tp_getattro */       0,
    /* tp_setattro */       0,
    /* tp_as_buffer */      0,
    /* tp_flags */          Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /* tp_doc */            "Collaborative filtering trainer model.",
    /* tp_traverse */       0,
    /* tp_clear */          0,
    /* tp_richcompare */    0,
    /* tp_weaklistoffset */ 0,
    /* tp_iter */           0,
    /* tp_iternext */       0,
    /* tp_methods */        model_methods,
    /* tp_members */        model_members,
    /* tp_getset */         0,
    /* tp_base */           0,
    /* tp_dict */           0,
    /* tp_descr_get */      0,
    /* tp_descr_set */      0,
    /* tp_dictoffset */     0,
    /* tp_init */           (initproc)model_init,
    /* tp_alloc */          0,
    /* tp_new */            model_new,
};

/*
  Module definition.
--------------------------------------------------------------------------------
 */

static PyModuleDef trainermodule = {
    PyModuleDef_HEAD_INIT,
    "trainer",
    "Collaborative filtering trainer module.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_trainer(void) 
{
    PyObject *module;

    ModelType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ModelType) < 0) {
        return NULL;
    }

    module = PyModule_Create(&trainermodule);
    if (module == NULL) {
        return NULL;
    }

    Py_INCREF(&ModelType);
    PyModule_AddObject(module, "Model", (PyObject*)&ModelType);

    return module;
}
