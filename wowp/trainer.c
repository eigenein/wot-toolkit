#include <stdlib.h>
#include <time.h>

#include <Python.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    int row_count;
    int column_count;
    int value_count;
    int *rows;
    int *columns;
    double *values;
    double base;
    double *row_bases;
    double *column_bases;
} Model;

/*
  Model initialization and deallocation.
--------------------------------------------------------------------------------
 */

static PyObject *
model_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Model *self = (Model*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->row_count = 0;
        self->column_count = 0;
        self->value_count = 0;
        self->rows = NULL; // row indexes
        self->columns = NULL; // column indexes
        self->values = NULL; // ratings
        self->base = 0.0; // base predictor
        self->row_bases = NULL; // row base predictors
        self->column_bases = NULL; // column base predictors
    }
    return (PyObject*)self;
}

int alloc_wrapper(size_t n, void **p) {
    *p = PyMem_RawMalloc(n);
    if (*p != NULL) {
        return 1;
    } else {
        PyErr_SetString(PyExc_MemoryError, "not enough memory");
        return 0;
    }
}

static int
model_init(Model *self, PyObject *args, PyObject *kwargs) {
    // Parse arguments.
    static char *kwlist[] = {"row_count", "column_count", "value_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iii", kwlist, &self->row_count, &self->column_count, &self->value_count)) {
        return -1;
    }
    // Allocate memory.
    if (
        !alloc_wrapper(self->row_count * sizeof(int), (void**)&self->rows) ||
        !alloc_wrapper(self->column_count * sizeof(int), (void**)&self->columns) ||
        !alloc_wrapper(self->value_count * sizeof(double), (void**)&self->values) ||
        !alloc_wrapper(self->row_count * sizeof(double), (void**)&self->row_bases) ||
        !alloc_wrapper(self->column_count * sizeof(double), (void**)&self->column_bases)
    ) {
        return -1;
    }
    srand(time(NULL));
    return 0;
}

static void
model_dealloc(Model *self) {
    PyMem_RawFree(self->rows);
    PyMem_RawFree(self->columns);
    PyMem_RawFree(self->values);
    PyMem_RawFree(self->row_bases);
    PyMem_RawFree(self->column_bases);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/*
  Model methods.
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

#define SWAP(array, i, j, temp_type) { temp_type temp = array[i]; array[i] = array[j]; array[j] = temp; }

double
rand_wrapper(double randomness) {
    return randomness * (1.0 * rand() / RAND_MAX - 0.5);
}

static PyObject *
model_prepare(Model *self, PyObject *args, PyObject *kwargs) {
    double randomness;
    int i;
    double sum = 0.0;
    // Parse arguments.
    static char *kwlist[] = {"randomness", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &randomness)) {
        return NULL;
    }
    // Compute base.
    for (i = 0; i < self->value_count; i++) {
        sum += self->values[i];
    }
    self->base = sum / self->value_count;
    // Randomize row bases.
    for (i = 0; i < self->row_count; i++) {
        self->row_bases[i] = rand_wrapper(randomness);
    }
    // Randomize column bases.
    for (i = 0; i < self->column_count; i++) {
        self->column_bases[i] = rand_wrapper(randomness);
    }

    Py_RETURN_NONE;
}

static PyObject *
model_shuffle(Model *self) {
    int i, j;

    for (i = self->value_count - 1; i != 0; i--) {
        j = rand() % (i + 1);
        SWAP(self->rows, i, j, int);
        SWAP(self->columns, i, j, int);
        SWAP(self->values, i, j, double);
    }

    Py_RETURN_NONE;
}


/*
  Model definition.
--------------------------------------------------------------------------------
 */

static PyMemberDef model_members[] = {
    {"row_count", T_LONG, offsetof(Model, row_count), 0, "Row count."},
    {"column_count", T_LONG, offsetof(Model, column_count), 0, "Column count."},
    {"value_count", T_LONG, offsetof(Model, value_count), 0, "Value count."},
    {"base", T_DOUBLE, offsetof(Model, base), 0, "Base."},
    {NULL}
};

static PyMethodDef model_methods[] = {
    {"set_value", (PyCFunction)model_set_value, METH_VARARGS | METH_KEYWORDS, "Sets value."},
    {"prepare", (PyCFunction)model_prepare, METH_VARARGS | METH_KEYWORDS, "Prepares model for training."},
    {"shuffle", (PyCFunction)model_shuffle, METH_NOARGS, "Shuffles values."},
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
