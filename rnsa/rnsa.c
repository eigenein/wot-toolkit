#include <math.h>

#include <Python.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    /* Row count. */                   unsigned long row_count;
    /* Column count. */                unsigned long column_count;
    /* Value count. */                 unsigned long value_count;
    /* Cluster count. */               unsigned long k;
    /* Points to column starts. */     unsigned long *indptr;
    /* Row indices. */                 unsigned long *indices;
    /* Corresponding values. */        float *values;
    /* Cluster centers. */             float *centroids;
    /* Used when moving centroids. */  float *new_centroids;
    /* Used when moving centroids. */  unsigned long *new_counter;
} Model;

/*
  Model initialization and deallocation.
--------------------------------------------------------------------------------
 */

static PyObject *
model_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Model *self = (Model*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->column_count = self->value_count = self->k = 0;
        self->indices = self->indptr = NULL;
        self->values = NULL;
        self->centroids = NULL;
        self->new_centroids = NULL;
        self->new_counter = NULL;
    }
    return (PyObject*)self;
}

static int
model_init(Model *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"row_count", "column_count", "value_count", "k", NULL};
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "kkkk", kwlist,
        &self->row_count, &self->column_count, &self->value_count, &self->k
    )) {
        return -1;
    }
    if (
        !(self->indptr = PyMem_RawMalloc(self->column_count * sizeof(*self->indptr) + sizeof(*self->indptr))) ||
        !(self->indices = PyMem_RawMalloc(self->value_count * sizeof(*self->indices))) ||
        !(self->values = PyMem_RawMalloc(self->value_count * sizeof(*self->values))) ||
        !(self->centroids = PyMem_RawMalloc(self->k * self->row_count * sizeof(*self->centroids))) ||
        !(self->new_centroids = PyMem_RawMalloc(self->k * self->row_count * sizeof(*self->new_centroids))) ||
        !(self->new_counter = PyMem_RawMalloc(self->k * self->row_count * sizeof(*self->new_counter)))
    ) {
        PyErr_NoMemory();
        return -1;
    }
    self->indptr[0] = 0ul;
    self->indptr[self->column_count] = self->value_count;
    return 0;
}

static void
model_dealloc(Model *self) {
    PyMem_RawFree(self->indptr);
    PyMem_RawFree(self->indices);
    PyMem_RawFree(self->values);
    PyMem_RawFree(self->centroids);
    PyMem_RawFree(self->new_centroids);
    PyMem_RawFree(self->new_counter);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/*
  Model getters.
--------------------------------------------------------------------------------
 */

static PyObject *
model_get_centroid(Model *self, PyObject *args, PyObject *kwargs) {
    unsigned long index;

    static char *kwlist[] = {"index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "k", kwlist, &index)) {
        return NULL;
    }

    float (*centroids)[self->k] = (float (*)[self->k])self->centroids;
    PyObject *centroid = PyTuple_New(self->row_count);
    for (unsigned long i = 0; i < self->row_count; i++) {
        PyTuple_SET_ITEM(centroid, i, Py_BuildValue("f", centroids[index][i]));
    }

    return centroid;
}

/*
  Model setters.
--------------------------------------------------------------------------------
 */

static PyObject *
model_set_indptr(Model *self, PyObject *args, PyObject *kwargs) {
    unsigned long j, index;

    static char *kwlist[] = {"j", "index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "kk", kwlist, &j, &index)) {
        return NULL;
    }

    if ((j == 0) || (j >= self->column_count)) {
        PyErr_SetObject(PyExc_ValueError, Py_BuildValue("k", j));
        return NULL;
    }

    self->indptr[j] = index;

    Py_RETURN_NONE;
}

static PyObject *
model_set_value(Model *self, PyObject *args, PyObject *kwargs) {
    unsigned long index;
    unsigned long row;
    float value;

    static char *kwlist[] = {"index", "row", "value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "kkf", kwlist, &index, &row, &value)) {
        return NULL;
    }

    if (index >= self->value_count) {
        PyErr_SetObject(PyExc_ValueError, Py_BuildValue("k", index));
        return NULL;
    }

    self->indices[index] = row;
    self->values[index] = value;

    Py_RETURN_NONE;
}

/*
  Helpers.
--------------------------------------------------------------------------------
 */

float avg(const unsigned long *indptr, const float *values, const unsigned long j) {
    float sum = 0.0f;
    for (unsigned long index = indptr[j]; index != indptr[j + 1]; index += 1) {
        sum += values[index];
    }
    return sum / (indptr[j + 1] - indptr[j]);
}

float w(
    const unsigned long *indptr,
    const unsigned long *indices,
    const float *values,
    const unsigned long j1,
    const unsigned long j2
) {
    const float avg_1 = avg(indptr, values, j1);
    const float avg_2 = avg(indptr, values, j2);

    float upper_sum = 0.0f, sum_squared_1 = 0.0f, sum_squared_2 = 0.0f;

    unsigned long ptr_1 = indptr[j1], ptr_2 = indptr[j2];

    while ((ptr_1 != indptr[j1 + 1]) && (ptr_2 != indptr[j2 + 1])) {
        if (indices[ptr_1] == indices[ptr_2]) {
            const float diff_1 = values[ptr_1] - avg_1;
            const float diff_2 = values[ptr_2] - avg_2;
            upper_sum += diff_1 * diff_2;
            sum_squared_1 += diff_1 * diff_1;
            sum_squared_2 += diff_2 * diff_2;
            ptr_1 += 1;
            ptr_2 += 1;
        } else if (indices[ptr_1] < indices[ptr_2]) {
            ptr_1 += 1;
        } else {
            ptr_2 += 1;
        }
    }

    return upper_sum / sqrt(sum_squared_1 * sum_squared_2);
}

unsigned long find_nearest_centroid(
    const unsigned long row_count,
    const unsigned long k,
    const unsigned long *indptr,
    const unsigned long *indices,
    const float *values,
    const float *self_centroids,
    const unsigned long j
) {
    const float (*centroids)[k] = (float (*)[k])self_centroids;

    unsigned long nearest_index = rand() % k;
    float lowest_w = INFINITY;

    for (unsigned long index = 0; index < k; index++) {
        float avg_centroid = 0.0f;
        for (unsigned long i = 0; i < row_count; i++) {
            avg_centroid += centroids[index][i];
        }
        avg_centroid /= row_count;

        float avg_j = avg(indptr, values, j);

        float upper_sum = 0.0f, sum_squared_1 = 0.0f, sum_squared_2 = 0.0f;
        for (unsigned long ptr = indptr[j]; ptr != indptr[j + 1]; ptr++) {
            const float diff_centroid = centroids[index][indices[ptr]] - avg_centroid;
            const float diff_j = values[ptr] - avg_j;
            upper_sum += diff_centroid * diff_j;
            sum_squared_1 += diff_centroid * diff_centroid;
            sum_squared_2 += diff_j * diff_j;
        }

        const float w = upper_sum / sqrt(sum_squared_1 * sum_squared_2);
        if ((w == w) && (w < lowest_w)) {
            nearest_index = index;
            lowest_w = w;
        }
    }

    return nearest_index;
}

/*
  Model methods.
--------------------------------------------------------------------------------
 */

static PyObject *
model_init_centroids(Model *self, PyObject *args, PyObject *kwargs) {
    float a, b;

    static char *kwlist[] = {"a", "b", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ff", kwlist, &a, &b)) {
        return NULL;
    }

    if (a > b) {
        PyErr_SetString(PyExc_ValueError, "a > b");
        return NULL;
    }

    float (*centroids)[self->k] = (float (*)[self->k])self->centroids;

    for (unsigned long index = 0; index < self->k; index++) {
        for (unsigned long i = 0; i < self->row_count; i++) {
            centroids[index][i] = a + (b - a) * (1.0f * rand() / RAND_MAX);
        }
    }

    Py_RETURN_NONE;
}

static PyObject *
model_step(Model *self, PyObject *args, PyObject *kwargs) {
    float (*new_centroids)[self->k] = (float (*)[self->k])self->new_centroids;
    float (*centroids)[self->k] = (float (*)[self->k])self->centroids;
    unsigned long (*new_counter)[self->k] = (unsigned long (*)[self->k])self->new_counter;

    for (unsigned long index = 0; index < self->k; index++) {
        for (unsigned long i = 0; i < self->row_count; i++) {
            new_centroids[index][i] = 0.0f;
            new_counter[index][i] = 0ul;
        }
    }
    // Cluster assignment step.
    for (unsigned long j = 0; j < self->column_count; j++) {
        const unsigned long index = find_nearest_centroid(
            self->row_count, self->k, self->indptr, self->indices, self->values, self->centroids, j);
        for (unsigned long ptr = self->indptr[j]; ptr != self->indptr[j + 1]; ptr++) {
            new_centroids[index][self->indices[ptr]] += self->values[ptr];
            new_counter[index][self->indices[ptr]] += 1;
        }
    }
    // Move centroids.
    for (unsigned long index = 0; index < self->k; index++) {
        for (unsigned long i = 0; i < self->row_count; i++) {
            if (new_counter[index][i] != 0) {
                centroids[index][i] = new_centroids[index][i] / new_counter[index][i];
            } else {
                centroids[index][i] = 0.0f;
            }
        }
    }

    Py_RETURN_NONE;
}

static PyObject *
model_cost(Model *self, PyObject *args, PyObject *kwargs) {
    // TODO: compute cost.
    Py_RETURN_NONE;
}

static PyObject *
model_avg(Model *self, PyObject *args, PyObject *kwargs) {
    unsigned long j;

    static char *kwlist[] = {"j", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "k", kwlist, &j)) {
        return NULL;
    }

    return Py_BuildValue("f", avg(self->indptr, self->values, j));
}

static PyObject *
model_w(Model *self, PyObject *args, PyObject *kwargs) {
    unsigned long j1, j2;

    static char *kwlist[] = {"j1", "j2", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "kk", kwlist, &j1, &j2)) {
        return NULL;
    }

    return Py_BuildValue("f", w(self->indptr, self->indices, self->values, j1, j2));
}

static PyObject *
model_find_nearest_centroid(Model *self, PyObject *args, PyObject *kwargs) {
    unsigned long j;

    static char *kwlist[] = {"j", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "k", kwlist, &j)) {
        return NULL;
    }

    return Py_BuildValue("f", find_nearest_centroid(
        self->row_count, self->k, self->indptr, self->indices, self->values, self->centroids, j));
}

/*
  Model definition.
--------------------------------------------------------------------------------
 */

static PyMemberDef model_members[] = {
    {"row_count", T_ULONG, offsetof(Model, row_count), 0, "Row count."},
    {"column_count", T_ULONG, offsetof(Model, column_count), 0, "Column count."},
    {"value_count", T_ULONG, offsetof(Model, value_count), 0, "Value count."},
    {"k", T_ULONG, offsetof(Model, k), 0, "Cluster count."},
    {NULL}
};

static PyMethodDef model_methods[] = {
    {"get_centroid", (PyCFunction)model_get_centroid, METH_VARARGS | METH_KEYWORDS, "Gets centroid coordinate."},
    {"set_indptr", (PyCFunction)model_set_indptr, METH_VARARGS | METH_KEYWORDS, "Sets column start index."},
    {"set_value", (PyCFunction)model_set_value, METH_VARARGS | METH_KEYWORDS, "Sets value at the specified row position."},
    {"init_centroids", (PyCFunction)model_init_centroids, METH_VARARGS | METH_KEYWORDS, "Randomly initializes centroids."},
    {"step", (PyCFunction)model_step, METH_VARARGS | METH_KEYWORDS, "Does k-means algorithm iteration."},
    {"cost", (PyCFunction)model_cost, METH_VARARGS | METH_KEYWORDS, "Computes current cost."},
    {"_avg", (PyCFunction)model_avg, METH_VARARGS | METH_KEYWORDS, "Computes average rating."},
    {"_w", (PyCFunction)model_w, METH_VARARGS | METH_KEYWORDS, "Computes correlation."},
    {"_find_nearest_centroid", (PyCFunction)model_find_nearest_centroid, METH_VARARGS | METH_KEYWORDS, "Finds nearest centroid."},
    {NULL}
};

static PyTypeObject ModelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /* tp_name */          "rnsa.Model",
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
    /* tp_doc */            "RNSA model.",
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

static PyModuleDef rnsa_module = {
    PyModuleDef_HEAD_INIT,
    "rnsa",
    "Refined Neighbor Selection Algorithm extension.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_rnsa(void) 
{
    PyObject *module;

    module = PyModule_Create(&rnsa_module);
    if (module == NULL) {
        return NULL;
    }

    ModelType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ModelType) < 0) {
        return NULL;
    }

    Py_INCREF(&ModelType);
    PyModule_AddObject(module, "Model", (PyObject*)&ModelType);

    return module;
}
