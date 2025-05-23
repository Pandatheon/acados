/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


#include <assert.h>
#include <stdlib.h>

#include "acados/utils/external_function_generic.h"
#include "acados/utils/mem.h"


/* general utilities for all external_function_* */

void external_function_opts_copy(external_function_opts *from, external_function_opts* to)
{
    to->external_workspace = from->external_workspace;
    to->with_global_data = from->with_global_data;
}

void external_function_opts_set_to_default(external_function_opts *opts)
{
    opts->external_workspace = false;
    opts->with_global_data = false;
}

size_t external_function_get_workspace_requirement_if_defined(external_function_generic *fun)
{
    if (fun == NULL)
        return 0;
    else
        return fun->get_external_workspace_requirement(fun);
}

void external_function_set_fun_workspace_if_defined(external_function_generic *fun, void *work_)
{
    if (fun != NULL)
        fun->set_external_workspace(fun, work_);
}



/************************************************
 * generic external parametric function
 ************************************************/

acados_size_t external_function_param_generic_struct_size()
{
    return sizeof(external_function_param_generic);
}



static void external_function_param_generic_set_param_sparse(void *self, int n_update,
                                                             int *idx, double *p)
{
    external_function_param_generic *fun = self;

    for (int ii = 0; ii < n_update; ii++)
    {
        fun->p[idx[ii]] = p[ii];
    }

    return;
}


acados_size_t external_function_param_generic_calculate_size(external_function_param_generic *fun, int np, external_function_opts *opts_)
{
    // wrapper as evaluate function
    fun->evaluate = &external_function_param_generic_wrapper;
    fun->get_external_workspace_requirement = external_function_param_generic_get_external_workspace_requirement;
    fun->set_external_workspace = external_function_param_generic_set_external_workspace;

    // set param function
    fun->get_nparam = &external_function_param_generic_get_nparam;
    fun->set_param = &external_function_param_generic_set_param;
    fun->set_param_sparse = &external_function_param_generic_set_param_sparse;

    // set number of parameters
    fun->np = np;

    // copy options
    external_function_opts_copy(opts_, &fun->opts);

    if (opts_->with_global_data)
    {
        printf("\nexternal_function_param_generic: option with_global_data not implemented!!!\n");
        exit(1);
    }

    acados_size_t size = 0;

    // doubles
    size += fun->np * sizeof(double); // p

    size += 8;  // align to double

    make_int_multiple_of(8, &size);

    return size;
}



void external_function_param_generic_assign(external_function_param_generic *fun, void *raw_memory)
{
    // save initial pointer to external memory
    fun->ptr_ext_mem = raw_memory;

    // char pointer for byte advances
    char *c_ptr = raw_memory;

    // align to double
    align_char_to(8, &c_ptr);

    // p
    assign_and_advance_double(fun->np, &fun->p, &c_ptr);

    assert((char *) raw_memory + external_function_param_generic_calculate_size(fun, fun->np, &fun->opts) >= c_ptr);

    return;
}



void external_function_param_generic_wrapper(void *self, ext_fun_arg_t *type_in, void **in, ext_fun_arg_t *type_out, void **out)
{
    // cast into external generic function
    external_function_param_generic *fun = self;

    fun->fun(in, out, fun->p);

    return;
}



void external_function_param_generic_get_nparam(void *self, int *np)
{
    // cast into external generic function
    external_function_param_generic *fun = self;

    *np = fun->np;

    return;
}



void external_function_param_generic_set_param(void *self, double *p)
{
    // cast into external generic function
    external_function_param_generic *fun = self;

    // set value for all parameters
    for (int ii = 0; ii < fun->np; ii++)
    {
        fun->p[ii] = p[ii];
    }

    return;
}


size_t external_function_param_generic_get_external_workspace_requirement(void *self)
{
    // external_function_param_generic *fun = self;
    return 0;
}

void external_function_param_generic_set_external_workspace(void *self, void *workspace)
{
    // do nothing
}


/************************************************
 * casadi utils
 ************************************************/

static int casadi_nnz(const int *sparsity)
{
    int nnz = 0;
    if (sparsity != NULL)
    {
        const int nrow = sparsity[0];
        const int ncol = sparsity[1];
        const int dense = sparsity[2];
        if (dense)
        {
            nnz = nrow * ncol;
        }
        else
        {
            const int *idxcol = sparsity + 2;
            for (int i = 0; i < ncol; ++i)
            {
                nnz += idxcol[i + 1] - idxcol[i];
            }
        }
    }

    return nnz;
}


static int casadi_is_dense(const int *sparsity)
{
    // returns true if casadi sparsity goes through all elements of the matrix as it would for a dense matrix;
    // otherwise false.
    int idx, jj;
    if (sparsity != NULL)
    {
        const int nrow = sparsity[0];
        const int ncol = sparsity[1];
        const int dense = sparsity[2];
        if ((nrow<=0 )| (ncol<=0))
            return 1;
        if (dense)
        {
            return 1;
        }
        else
        {
            // check for casadi fake sparse
            const int *idxcol = sparsity + 2;
            const int *row = sparsity + ncol + 3;
            for (jj = 0; jj < ncol; jj++)
            {
                if (idxcol[jj + 1] - idxcol[jj] != nrow)
                {
                    // printf("casadi_is_dense: REAL sparse!\n");
                    // printf("idxcol[jj+1] = %d idxcol[jj] %d nrow %d\n", idxcol[jj + 1], idxcol[jj], nrow);
                    return 0;
                }
                for (idx = idxcol[jj]; idx != idxcol[jj + 1]; idx++)
                {
                    if (row[idx]+jj*nrow != idx)
                    {
                        // printf("casadi_is_dense: REAL sparse!\n");
                        // printf("row[idx] %d != idx = %d, nrow %d ncol %d, jj %d\n", row[idx], idx, nrow, ncol, jj);
                        return 0;
                    }
                }
            }
        }
    }
    // printf("casadi_is_dense: real DENSE!\n");

    return 1;
}





static void d_cvt_casadi_to_colmaj(double *in, int *sparsity_in, double *out, int is_dense)
{
    int ii, jj, idx;

    int nrow = sparsity_in[0];
    int ncol = sparsity_in[1];

    if ((nrow<=0 )| (ncol<=0))
        return;

    if (is_dense)
    {
        // printf("d_cvt_casadi_to_colmaj: dense\n");
        for (ii = 0; ii < ncol * nrow; ii++) out[ii] = in[ii];
    }
    else
    {
        // printf("d_cvt_casadi_to_colmaj: sparse\n");
        double *ptr = in;
        int *idxcol = sparsity_in + 2;
        int *row = sparsity_in + ncol + 3;
        // Fill with zeros
        for (jj = 0; jj < ncol; jj++)
            for (ii = 0; ii < nrow; ii++) out[ii + jj * nrow] = 0.0;
        // Copy nonzeros
        for (jj = 0; jj < ncol; jj++)
        {
            for (idx = idxcol[jj]; idx != idxcol[jj + 1]; idx++)
            {
                out[row[idx] + jj * nrow] = ptr[0];
                ptr++;
            }
        }
    }

    return;
}



static void d_cvt_colmaj_to_casadi(double *in, double *out, int *sparsity_out, int is_dense)
{
    int ii, jj, idx;

    int nrow = sparsity_out[0];
    int ncol = sparsity_out[1];

    if ((nrow<=0 )| (ncol<=0))
        return;

    if (is_dense)
    {
        // printf("d_cvt_colmaj_to_casadi: dense\n");
        for (ii = 0; ii < ncol * nrow; ii++) out[ii] = in[ii];
    }
    else
    {
        // printf("d_cvt_colmaj_to_casadi: sparse\n");
        double *ptr = out;
        int *idxcol = sparsity_out + 2;
        int *row = sparsity_out + ncol + 3;
        // Copy nonzeros
        for (jj = 0; jj < ncol; jj++)
        {
            for (idx = idxcol[jj]; idx != idxcol[jj + 1]; idx++)
            {
                ptr[0] = in[row[idx] + jj * nrow];
                ptr++;
            }
        }
    }

    return;
}



static void d_cvt_casadi_to_dmat(double *in, int *sparsity_in, struct blasfeo_dmat *out, int is_dense)
{
    int jj, idx;

    int nrow = sparsity_in[0];
    int ncol = sparsity_in[1];

    if ((nrow<=0 )| (ncol<=0))
        return;

    if (is_dense)
    {
        blasfeo_pack_dmat(nrow, ncol, in, nrow, out, 0, 0);
    }
    else
    {
        double *ptr = in;
        int *idxcol = sparsity_in + 2;
        int *row = sparsity_in + ncol + 3;
        // Fill with zeros
        blasfeo_dgese(nrow, ncol, 0.0, out, 0, 0);
        // Copy nonzeros
        for (jj = 0; jj < ncol; jj++)
        {
            for (idx = idxcol[jj]; idx != idxcol[jj + 1]; idx++)
            {
                BLASFEO_DMATEL(out, row[idx], jj) = ptr[0];
                ptr++;
            }
        }
    }

    return;
}



static void d_cvt_dmat_to_casadi(struct blasfeo_dmat *in, double *out, int *sparsity_out, int is_dense)
{
    int jj, idx;

    int nrow = sparsity_out[0];
    int ncol = sparsity_out[1];

    if ((nrow<=0 )| (ncol<=0))
        return;

    if (is_dense)
    {
        blasfeo_unpack_dmat(nrow, ncol, in, 0, 0, out, nrow);
    }
    else
    {
        double *ptr = out;
        int *idxcol = sparsity_out + 2;
        int *row = sparsity_out + ncol + 3;
        // Copy nonzeros
        for (jj = 0; jj < ncol; jj++)
        {
            for (idx = idxcol[jj]; idx != idxcol[jj + 1]; idx++)
            {
                ptr[0] = BLASFEO_DMATEL(in, row[idx], jj);
                ptr++;
            }
        }
    }

    return;
}



static void d_cvt_casadi_to_dvec(double *in, int *sparsity_in, struct blasfeo_dvec *out, int is_dense)
{
    int idx;

    if (!((sparsity_in[1] == 1) | (sparsity_in[0] == 0) | (sparsity_in[1] == 0)))
    {
        printf("\nd_cvt_casadi_to_dvec: expected column vector or empty vector. Exiting.\n\n");
        exit(1);
    }

    int n = sparsity_in[0];

    if (n<=0)
        return;

    if (is_dense)
    {
        blasfeo_pack_dvec(n, in, 1, out, 0);
    }
    else
    {
        double *ptr = in;
        int *idxcol = sparsity_in + 2;
        int *row = sparsity_in + 1 + 3;
        // Fill with zeros
        blasfeo_dvecse(n, 0.0, out, 0);
        // Copy nonzeros
        for (idx = idxcol[0]; idx != idxcol[1]; idx++)
        {
            BLASFEO_DVECEL(out, row[idx]) = ptr[0];
            ptr++;
        }
    }

    return;
}



static void d_cvt_dvec_to_casadi(struct blasfeo_dvec *in, double *out, int *sparsity_out, int is_dense)
{
    int idx;

    if (!((sparsity_out[1] == 1) | (sparsity_out[0] == 0) | (sparsity_out[1] == 0)))
    {
        printf("\nd_cvt_dvec_to_casadi: expected column vector or empty vector. Exiting.\n\n");
        exit(1);
    }
    int n = sparsity_out[0];

    if (n<=0)
        return;

    if (is_dense)
    {
        blasfeo_unpack_dvec(n, in, 0, out, 1);
    }
    else
    {
        double *ptr = out;
        int *idxcol = sparsity_out + 2;
        int *row = sparsity_out + 1 + 3;
        // Copy nonzeros
        for (idx = idxcol[0]; idx != idxcol[1]; idx++)
        {
            ptr[0] = BLASFEO_DVECEL(in, row[idx]);
            ptr++;
        }
    }

    return;
}



static void d_cvt_casadi_to_colmaj_args(double *in, int *sparsity_in, struct colmaj_args *out, int is_dense)
{
    int ii, jj, idx;

    int nrow = sparsity_in[0];
    int ncol = sparsity_in[1];

    if ((nrow<=0 )| (ncol<=0))
        return;

    double *A = out->A;
    int lda = out->lda;

    if (is_dense)
    {
        for (ii = 0; ii < ncol; ii++)
            for (jj = 0; jj < nrow; jj++) A[ii + jj * lda] = in[ii + ncol * jj];
    }
    else
    {
        double *ptr = in;
        int *idxcol = sparsity_in + 2;
        int *row = sparsity_in + ncol + 3;
        // Fill with zeros
        for (jj = 0; jj < ncol; jj++)
            for (ii = 0; ii < nrow; ii++) A[ii + jj * lda] = 0.0;
        // Copy nonzeros
        for (jj = 0; jj < ncol; jj++)
        {
            for (idx = idxcol[jj]; idx != idxcol[jj + 1]; idx++)
            {
                A[row[idx] + jj * lda] = ptr[0];
                ptr++;
            }
        }
    }

    return;
}



static void d_cvt_colmaj_args_to_casadi(struct colmaj_args *in, double *out, int *sparsity_out, int is_dense)
{
    int ii, jj, idx;

    int nrow = sparsity_out[0];
    int ncol = sparsity_out[1];

    if ((nrow<=0 )| (ncol<=0))
        return;

    double *A = in->A;
    int lda = in->lda;

    if (is_dense)
    {
        for (ii = 0; ii < ncol; ii++)
            for (jj = 0; jj < nrow; jj++) out[ii + ncol * jj] = A[ii + jj * lda];
    }
    else
    {
        double *ptr = out;
        int *idxcol = sparsity_out + 2;
        int *row = sparsity_out + ncol + 3;
        // Copy nonzeros
        for (jj = 0; jj < ncol; jj++)
        {
            for (idx = idxcol[jj]; idx != idxcol[jj + 1]; idx++)
            {
                ptr[0] = A[row[idx] + jj * lda];
                ptr++;
            }
        }
    }

    return;
}



static void d_cvt_casadi_to_dmat_args(double *in, int *sparsity_in, struct blasfeo_dmat_args *out, int is_dense)
{
    int jj, idx;

    int nrow = sparsity_in[0];
    int ncol = sparsity_in[1];

    if ((nrow<=0 )| (ncol<=0))
        return;

    struct blasfeo_dmat *A = out->A;
    int ai = out->ai;
    int aj = out->aj;

    if (is_dense)
    {
        blasfeo_pack_dmat(nrow, ncol, in, nrow, A, ai, aj);
    }
    else
    {
        double *ptr = in;
        int *idxcol = sparsity_in + 2;
        int *row = sparsity_in + ncol + 3;
        // Fill with zeros
        blasfeo_dgese(nrow, ncol, 0.0, A, ai, aj);
        // Copy nonzeros
        for (jj = 0; jj < ncol; jj++)
        {
            for (idx = idxcol[jj]; idx != idxcol[jj + 1]; idx++)
            {
                BLASFEO_DMATEL(A, ai + row[idx], aj + jj) = ptr[0];
                ptr++;
            }
        }
    }

    return;
}



static void d_cvt_dmat_args_to_casadi(struct blasfeo_dmat_args *in, double *out, int *sparsity_out, int is_dense)
{
    int jj, idx;

    int nrow = sparsity_out[0];
    int ncol = sparsity_out[1];

    if ((nrow<=0 )| (ncol<=0))
        return;

    struct blasfeo_dmat *A = in->A;
    int ai = in->ai;
    int aj = in->aj;

    if (is_dense)
    {
        blasfeo_unpack_dmat(nrow, ncol, A, ai, aj, out, nrow);
    }
    else
    {
        double *ptr = out;
        int *idxcol = sparsity_out + 2;
        int *row = sparsity_out + ncol + 3;
        // Copy nonzeros
        for (jj = 0; jj < ncol; jj++)
        {
            for (idx = idxcol[jj]; idx != idxcol[jj + 1]; idx++)
            {
                ptr[0] = BLASFEO_DMATEL(A, ai + row[idx], aj + jj);
                ptr++;
            }
        }
    }

    return;
}



static void d_cvt_casadi_to_dvec_args(double *in, int *sparsity_in, struct blasfeo_dvec_args *out, int is_dense)
{
    int idx;

    if (!((sparsity_in[1] == 1) | (sparsity_in[0] == 0) | (sparsity_in[1] == 0)))
    {
        printf("\nd_cvt_casadi_to_dvec_args: expected column vector or empty vector. Exiting.\n\n");
        exit(1);
    }

    int n = sparsity_in[0];

    if (n<=0)
        return;

    struct blasfeo_dvec *x = out->x;
    int xi = out->xi;

    if (is_dense)
    {
        blasfeo_pack_dvec(n, in, 1, x, xi);
    }
    else
    {
        double *ptr = in;
        int *idxcol = sparsity_in + 2;
        int *row = sparsity_in + 1 + 3;
        // Fill with zeros
        blasfeo_dvecse(n, 0.0, x, xi);
        // Copy nonzeros
        for (idx = idxcol[0]; idx != idxcol[1]; idx++)
        {
            BLASFEO_DVECEL(x, xi + row[idx]) = ptr[0];
            ptr++;
        }
    }

    return;
}



static void d_cvt_dvec_args_to_casadi(struct blasfeo_dvec_args *in, double *out, int *sparsity_out, int is_dense)
{
    int idx;

    if (!((sparsity_out[1] == 1) | (sparsity_out[0] == 0) | (sparsity_out[1] == 0)))
    {
        printf("\nd_cvt_dvec_args_to_casadi: expected column vector or empty vector. Exiting.\n\n");
        exit(1);
    }
    int n = sparsity_out[0];

    if (n<=0)
        return;

    struct blasfeo_dvec *x = in->x;
    int xi = in->xi;

    if (is_dense)
    {
        blasfeo_unpack_dvec(n, x, xi, out, 1);
    }
    else
    {
        double *ptr = out;
        int *idxcol = sparsity_out + 2;
        int *row = sparsity_out + 1 + 3;
        // Copy nonzeros
        for (idx = idxcol[0]; idx != idxcol[1]; idx++)
        {
            ptr[0] = BLASFEO_DVECEL(x, xi + row[idx]);
            ptr++;
        }
    }

    return;
}


static int d_cvt_casadi_to_ext_fun_arg(ext_fun_arg_t type, double *in, int *sparsity, void *out, int is_dense)
{
    switch (type)
    {
        case COLMAJ:
            d_cvt_casadi_to_colmaj(in, sparsity, out, is_dense);
            break;

        case BLASFEO_DMAT:
            d_cvt_casadi_to_dmat(in, sparsity, out, is_dense);
            break;

        case BLASFEO_DVEC:
            d_cvt_casadi_to_dvec(in, sparsity, out, is_dense);
            break;
        case COLMAJ_ARGS:
            d_cvt_casadi_to_colmaj_args(in, sparsity, out, is_dense);
            break;

        case BLASFEO_DMAT_ARGS:
            d_cvt_casadi_to_dmat_args(in, sparsity, out, is_dense);
            break;

        case BLASFEO_DVEC_ARGS:
            d_cvt_casadi_to_dvec_args(in, sparsity, out, is_dense);
            break;

        case IGNORE_ARGUMENT:
            // do nothing
            break;

        default:
            return 1;

    }
    return 0;
}

static int d_cvt_ext_fun_arg_to_casadi(ext_fun_arg_t type, void *in, double *out, int *sparsity, int is_dense)
{
    switch (type)
    {
        case COLMAJ:
            d_cvt_colmaj_to_casadi(in, out, sparsity, is_dense);
            break;

        case BLASFEO_DMAT:
            d_cvt_dmat_to_casadi(in, out, sparsity, is_dense);
            break;

        case BLASFEO_DVEC:
            d_cvt_dvec_to_casadi(in, out, sparsity, is_dense);
            break;

        case COLMAJ_ARGS:
            d_cvt_colmaj_args_to_casadi(in, out, sparsity, is_dense);
            break;

        case BLASFEO_DMAT_ARGS:
            d_cvt_dmat_args_to_casadi(in, out, sparsity, is_dense);
            break;

        case BLASFEO_DVEC_ARGS:
            d_cvt_dvec_args_to_casadi(in, out, sparsity, is_dense);
            break;

        case IGNORE_ARGUMENT:
            // do nothing
            break;

        default:
            return 1;
    }
    return 0;
}




/************************************************
 * casadi external function
 ************************************************/

acados_size_t external_function_casadi_struct_size()
{
    return sizeof(external_function_casadi);
}


acados_size_t external_function_casadi_calculate_size(external_function_casadi *fun, external_function_opts *opts_)
{
    // casadi wrapper as evaluate
    fun->evaluate = &external_function_casadi_wrapper;
    fun->get_external_workspace_requirement = external_function_casadi_get_external_workspace_requirement;
    fun->set_external_workspace = external_function_casadi_set_external_workspace;

    int ii;

    fun->casadi_work(&fun->args_num, &fun->res_num, &fun->int_work_size, &fun->float_work_size);

    fun->in_num = fun->casadi_n_in();
    fun->out_num = fun->casadi_n_out();

    // args
    fun->args_size_tot = 0;
    for (ii = 0; ii < fun->args_num; ii++)
        fun->args_size_tot += casadi_nnz(fun->casadi_sparsity_in(ii));

    // res
    fun->res_size_tot = 0;
    for (ii = 0; ii < fun->res_num; ii++)
        fun->res_size_tot += casadi_nnz(fun->casadi_sparsity_out(ii));

    // copy options
    external_function_opts_copy(opts_, &fun->opts);

    if (opts_->with_global_data)
    {
        printf("\nexternal_function_casadi: option with_global_data not implemented!!!\n");
        exit(1);
    }

    acados_size_t size = 0;

    // double pointers
    size += fun->args_num * sizeof(double *);  // args
    size += fun->res_num * sizeof(double *);   // res

    // ints
    size += 2 * fun->args_num * sizeof(int);  // args_size, args_dense
    size += 2 * fun->res_num * sizeof(int);   // res_size, res_dense
    size += fun->int_work_size * sizeof(int);   // int_work

    // doubles
    size += fun->args_size_tot * sizeof(double);  // args
    size += fun->res_size_tot * sizeof(double);   // res
    // float_work
    if (!fun->opts.external_workspace)
    {
        size += fun->float_work_size * sizeof(double);
    }

    size += 8;  // initial align
    size += 8;  // align to double

    make_int_multiple_of(8, &size);

    return size;
}



void external_function_casadi_assign(external_function_casadi *fun, void *raw_memory)
{
    int ii;

    // save initial pointer to external memory
    fun->ptr_ext_mem = raw_memory;

    // char pointer for byte advances
    char *c_ptr = raw_memory;

    // double pointers

    // initial align
    align_char_to(8, &c_ptr);

    // args
    assign_and_advance_double_ptrs(fun->args_num, &fun->args, &c_ptr);
    // res
    assign_and_advance_double_ptrs(fun->res_num, &fun->res, &c_ptr);

    // args_size, args_dense
    assign_and_advance_int(fun->args_num, &fun->args_size, &c_ptr);
    assign_and_advance_int(fun->args_num, &fun->args_dense, &c_ptr);
    for (ii = 0; ii < fun->args_num; ii++)
    {
        fun->args_size[ii] = casadi_nnz(fun->casadi_sparsity_in(ii));
        fun->args_dense[ii] = casadi_is_dense(fun->casadi_sparsity_in(ii));
    }
    // res_size, res_dense
    assign_and_advance_int(fun->res_num, &fun->res_size, &c_ptr);
    assign_and_advance_int(fun->res_num, &fun->res_dense, &c_ptr);
    for (ii = 0; ii < fun->res_num; ii++)
    {
        fun->res_size[ii] = casadi_nnz(fun->casadi_sparsity_out(ii));
        fun->res_dense[ii] = casadi_is_dense(fun->casadi_sparsity_out(ii));
    }
    // int_work
    assign_and_advance_int(fun->int_work_size, &fun->int_work, &c_ptr);

    // align to double
    align_char_to(8, &c_ptr);

    // args
    for (ii = 0; ii < fun->args_num; ii++)
        assign_and_advance_double(fun->args_size[ii], &fun->args[ii], &c_ptr);
    // res
    for (ii = 0; ii < fun->res_num; ii++)
        assign_and_advance_double(fun->res_size[ii], &fun->res[ii], &c_ptr);
    // float_work
    if (!fun->opts.external_workspace)
    {
        assign_and_advance_double(fun->float_work_size, &fun->float_work, &c_ptr);
    }

    assert((char *) raw_memory + external_function_casadi_calculate_size(fun, &fun->opts) >= c_ptr);

    return;
}



void external_function_casadi_wrapper(void *self, ext_fun_arg_t *type_in, void **in,
                                      ext_fun_arg_t *type_out, void **out)
{
    // cast into external casadi function
    external_function_casadi *fun = self;

    int ii;
    int status = 0;

    // in as args
    for (ii = 0; ii < fun->in_num; ii++)
    {
        status = d_cvt_ext_fun_arg_to_casadi(type_in[ii], in[ii], (double *) fun->args[ii],
                                    (int *) fun->casadi_sparsity_in(ii), fun->args_dense[ii]);
        if (status)
        {
            printf("\nexternal_function_casadi_wrapper: Unknown external function argument type %d for input %d\n\n", type_in[ii], ii);
            exit(1);
        }
    }

    // call casadi function
    fun->casadi_fun((const double **) fun->args, fun->res, fun->int_work, fun->float_work, NULL);

    for (ii = 0; ii < fun->out_num; ii++)
    {
        status = d_cvt_casadi_to_ext_fun_arg(type_out[ii], (double *) fun->res[ii], (int *) fun->casadi_sparsity_out(ii),
                                     out[ii], fun->res_dense[ii]);
        if (status)
        {
            printf("\nexternal_function_casadi_wrapper: Unknown external function argument type %d for output %d\n\n", type_out[ii], ii);
            exit(1);
        }
    }

    return;
}


size_t external_function_casadi_get_external_workspace_requirement(void *self)
{
    external_function_casadi *fun = self;
    if (fun->opts.external_workspace)
        return fun->float_work_size * sizeof(double);
    else
        return 0;
}

void external_function_casadi_set_external_workspace(void *self, void *workspace)
{
    external_function_casadi *fun = self;
    if (fun->opts.external_workspace)
        fun->float_work = workspace;
}


/************************************************
 * casadi external parametric function
 ************************************************/

acados_size_t external_function_param_casadi_struct_size()
{
    return sizeof(external_function_param_casadi);
}


static void external_function_param_casadi_set_param(void *self, double *p)
{
    external_function_param_casadi *fun = self;

    // set value for all parameters
    int* sparsity = (int *) fun->casadi_sparsity_in(fun->idx_in_p);
    d_cvt_colmaj_to_casadi(p, (double *) fun->args[fun->idx_in_p],
                            sparsity, fun->args_dense[fun->idx_in_p]);

    return;
}


static void external_function_param_casadi_set_param_sparse(void *self, int n_update,
                                                            int *idx_p_update, double *p)
{
    external_function_param_casadi *fun = self;

    if (fun->args_dense[fun->idx_in_p])
    {
        for (int ii = 0; ii < n_update; ii++)
        {
            fun->args[fun->idx_in_p][idx_p_update[ii]] = p[ii];
        }
    }
    else
    {
        printf("\nexternal_function_param_casadi_set_param_sparse: sparse parameter update for sparse parameter vector not supported!\n");
        exit(1);
    }

    return;
}


acados_size_t external_function_param_casadi_calculate_size(external_function_param_casadi *fun, int np, external_function_opts *opts_)
{
    int ii;

    // casadi wrapper as evaluate function
    fun->evaluate = &external_function_param_casadi_wrapper;
    fun->get_external_workspace_requirement = external_function_param_casadi_get_external_workspace_requirement;
    fun->set_external_workspace = external_function_param_casadi_set_external_workspace;

    // set param function
    fun->get_nparam = &external_function_param_casadi_get_nparam;
    fun->set_param = &external_function_param_casadi_set_param;
    fun->set_param_sparse = &external_function_param_casadi_set_param_sparse;

    // copy options
    external_function_opts_copy(opts_, &fun->opts);

    if (opts_->with_global_data)
    {
        printf("\nexternal_function_param_casadi: option with_global_data not implemented!!!\n");
        exit(1);
    }

    // set number of parameters
    fun->np = np;

    // compute workspace sizes
    fun->casadi_work(&fun->args_num, &fun->res_num, &fun->int_work_size, &fun->float_work_size);

    fun->in_num = fun->casadi_n_in();
    fun->out_num = fun->casadi_n_out();

    // parameter is last input
    fun->idx_in_p = fun->in_num-1;

    // args
    fun->args_size_tot = 0;
    for (ii = 0; ii < fun->args_num; ii++)
        fun->args_size_tot += casadi_nnz(fun->casadi_sparsity_in(ii));

    // res
    fun->res_size_tot = 0;
    for (ii = 0; ii < fun->res_num; ii++)
        fun->res_size_tot += casadi_nnz(fun->casadi_sparsity_out(ii));

    acados_size_t size = 0;

    // double pointers
    size += fun->args_num * sizeof(double *);  // args
    size += fun->res_num * sizeof(double *);   // res

    // ints
    size += 2 * fun->args_num * sizeof(int);  // args_size, args_dense
    size += 2 * fun->res_num * sizeof(int);   // res_size, res_dense
    size += fun->int_work_size * sizeof(int);   // int_work

    // doubles
    size += fun->args_size_tot * sizeof(double);  // args
    size += fun->res_size_tot * sizeof(double);   // res
    // float_work
    if (!fun->opts.external_workspace)
    {
        size += fun->float_work_size * sizeof(double);
    }

    size += 8;  // initial align
    size += 8;  // align to double

    make_int_multiple_of(8, &size);

    return size;
}



void external_function_param_casadi_assign(external_function_param_casadi *fun, void *raw_memory)
{
    int ii;

    // save initial pointer to external memory
    fun->ptr_ext_mem = raw_memory;

    // char pointer for byte advances
    char *c_ptr = raw_memory;

    // double pointers

    // initial align
    align_char_to(8, &c_ptr);

    // args
    assign_and_advance_double_ptrs(fun->args_num, &fun->args, &c_ptr);
    // res
    assign_and_advance_double_ptrs(fun->res_num, &fun->res, &c_ptr);

    // args_size, args_dense
    assign_and_advance_int(fun->args_num, &fun->args_size, &c_ptr);
    assign_and_advance_int(fun->args_num, &fun->args_dense, &c_ptr);
    for (ii = 0; ii < fun->args_num; ii++)
    {
        fun->args_size[ii] = casadi_nnz(fun->casadi_sparsity_in(ii));
        fun->args_dense[ii] = casadi_is_dense(fun->casadi_sparsity_in(ii));
    }
    // res_size, res_dense
    assign_and_advance_int(fun->res_num, &fun->res_size, &c_ptr);
    assign_and_advance_int(fun->res_num, &fun->res_dense, &c_ptr);
    for (ii = 0; ii < fun->res_num; ii++)
    {
        fun->res_size[ii] = casadi_nnz(fun->casadi_sparsity_out(ii));
        fun->res_dense[ii] = casadi_is_dense(fun->casadi_sparsity_out(ii));
    }
    // int_work
    assign_and_advance_int(fun->int_work_size, &fun->int_work, &c_ptr);

    // align to double
    align_char_to(8, &c_ptr);

    // args
    for (ii = 0; ii < fun->args_num; ii++)
        assign_and_advance_double(fun->args_size[ii], &fun->args[ii], &c_ptr);
    // res
    for (ii = 0; ii < fun->res_num; ii++)
        assign_and_advance_double(fun->res_size[ii], &fun->res[ii], &c_ptr);
    // float_work
    if (!fun->opts.external_workspace)
    {
        assign_and_advance_double(fun->float_work_size, &fun->float_work, &c_ptr);
    }

    assert((char *) raw_memory + external_function_param_casadi_calculate_size(fun, fun->np, &fun->opts) >=
           c_ptr);

    return;
}



void external_function_param_casadi_wrapper(void *self, ext_fun_arg_t *type_in, void **in,
                                            ext_fun_arg_t *type_out, void **out)
{
    // cast into external casadi function
    external_function_param_casadi *fun = self;
    int ii;
    int status = 0;
    // in as args
    for (ii = 0; ii < fun->in_num; ii++)
    {
        // skip parameter argument
        if (ii != fun->idx_in_p)
        {
            status = d_cvt_ext_fun_arg_to_casadi(type_in[ii], in[ii], (double *) fun->args[ii],
                                        (int *) fun->casadi_sparsity_in(ii), fun->args_dense[ii]);
        }
        if (status)
        {
            printf("\nexternal_function_param_casadi_wrapper: Unknown external function argument type %d for input %d\n\n", type_in[ii], ii);
            exit(1);
        }
    }
    // parameters are last argument and set via external_function_param_casadi_set_param

    // call casadi function
    fun->casadi_fun((const double **) fun->args, fun->res, fun->int_work, fun->float_work, NULL);

    for (ii = 0; ii < fun->out_num; ii++)
    {
        status = d_cvt_casadi_to_ext_fun_arg(type_out[ii], (double *) fun->res[ii], (int *) fun->casadi_sparsity_out(ii),
                                     out[ii], fun->res_dense[ii]);
        if (status)
        {
            printf("\nexternal_function_param_casadi_wrapper: Unknown external function argument type %d for output %d\n\n", type_out[ii], ii);
            exit(1);
        }
    }

    return;
}



void external_function_param_casadi_get_nparam(void *self, int *np)
{
    external_function_param_casadi *fun = self;

    *np = fun->np;

    return;
}



size_t external_function_param_casadi_get_external_workspace_requirement(void *self)
{
    external_function_param_casadi *fun = self;
    if (fun->opts.external_workspace)
        return fun->float_work_size * sizeof(double);
    else
        return 0;
}

void external_function_param_casadi_set_external_workspace(void *self, void *workspace)
{
    external_function_param_casadi *fun = self;
    if (fun->opts.external_workspace)
        fun->float_work = workspace;
}




/************************************************
 * generic external parametric function
 ************************************************/

acados_size_t external_function_external_param_generic_struct_size()
{
    return sizeof(external_function_external_param_generic);
}


static void external_function_external_param_generic_set_param_pointer(void *self, double *p)
{
    external_function_external_param_generic *fun = self;

    fun->p = p;
    fun->param_mem_is_set = true;

    return;
}

static void external_function_external_param_generic_set_global_data_pointer(void *self, double *global_data)
{
    // external_function_external_param_generic *fun = self;
    printf("external_function_external_param_generic_set_global_data_pointer: not implemented\n");
    exit(1);

    return;
}


acados_size_t external_function_external_param_generic_calculate_size(external_function_external_param_generic *fun, external_function_opts *opts_)
{
    // wrapper as evaluate function
    fun->evaluate = &external_function_external_param_generic_wrapper;
    fun->get_external_workspace_requirement = external_function_external_param_generic_get_external_workspace_requirement;
    fun->set_external_workspace = external_function_external_param_generic_set_external_workspace;

    // set param function
    fun->set_param_pointer = &external_function_external_param_generic_set_param_pointer;
    fun->set_global_data_pointer = &external_function_external_param_generic_set_global_data_pointer;

    // set flags
    fun->param_mem_is_set = false;
    // fun->global_data_ptr_is_set = false;

    // copy options
    external_function_opts_copy(opts_, &fun->opts);

    acados_size_t size = 0;

    make_int_multiple_of(8, &size);

    return size;
}



void external_function_external_param_generic_assign(external_function_external_param_generic *fun, void *raw_memory)
{
    // save initial pointer to external memory
    fun->ptr_ext_mem = raw_memory;

    // char pointer for byte advances
    // char *c_ptr = raw_memory;

    assert((char *) raw_memory + external_function_external_param_generic_calculate_size(fun, &fun->opts) >= (char *) raw_memory);

    return;
}



void external_function_external_param_generic_wrapper(void *self, ext_fun_arg_t *type_in, void **in, ext_fun_arg_t *type_out, void **out)
{
    // cast into external generic function
    external_function_external_param_generic *fun = self;

    fun->fun(in, out, fun->p);

    return;
}



size_t external_function_external_param_generic_get_external_workspace_requirement(void *self)
{
    // external_function_external_param_generic *fun = self;
    return 0;
}

void external_function_external_param_generic_set_external_workspace(void *self, void *workspace)
{
    // do nothing
}



/************************************************
 * external_function_external_param_casadi
 ************************************************/

acados_size_t external_function_external_param_casadi_struct_size()
{
    return sizeof(external_function_external_param_casadi);
}


static void external_function_external_param_casadi_set_param_pointer(void *self, double *p)
{
    external_function_external_param_casadi *fun = self;

    if (!fun->args_dense[fun->idx_in_p])
    {
        printf("\nexternal_function_external_param_casadi_set_param_pointer: sparse parameter not supported!\n");
        exit(1);
    }
    fun->args[fun->idx_in_p] = p;
    fun->param_mem_is_set = true;

    return;
}

static void external_function_external_param_casadi_set_global_data_pointer(void *self, double *global_data)
{
    external_function_external_param_casadi *fun = self;

    if (!fun->args_dense[fun->idx_in_global_data])
    {
        // NOTE: sparsity is removed in MATLAB/Python interface
        printf("\nexternal_function_external_param_casadi_set_global_data_pointer: sparse global_data not supported!\n");
        exit(1);
    }
    if (!fun->opts.with_global_data)
    {
        printf("\nexternal_function_external_param_casadi_set_global_data_pointer: not supported, as option opts.with_global_data is false.\n");
        exit(1);
    }
    fun->args[fun->idx_in_global_data] = global_data;
    fun->global_data_ptr_is_set = true;

    return;
}



acados_size_t external_function_external_param_casadi_calculate_size(external_function_external_param_casadi *fun, external_function_opts *opts_)
{
    int ii;

    // casadi wrapper as evaluate function
    fun->evaluate = &external_function_external_param_casadi_wrapper;
    fun->get_external_workspace_requirement = external_function_external_param_casadi_get_external_workspace_requirement;
    fun->set_external_workspace = external_function_external_param_casadi_set_external_workspace;

    // set param function
    fun->set_param_pointer = &external_function_external_param_casadi_set_param_pointer;
    fun->set_global_data_pointer = &external_function_external_param_casadi_set_global_data_pointer;

    fun->casadi_work(&fun->args_num, &fun->res_num, &fun->int_work_size, &fun->float_work_size);

    fun->in_num = fun->casadi_n_in();
    fun->out_num = fun->casadi_n_out();

    // copy options
    external_function_opts_copy(opts_, &fun->opts);

    // parameter indices
    if (fun->opts.with_global_data)
    {
        fun->idx_in_p = fun->in_num-2;
        fun->idx_in_global_data = fun->in_num-1;
    }
    else
    {
        fun->idx_in_p = fun->in_num-1;
        fun->idx_in_global_data = -1;
    }

    // args
    fun->args_size_tot = 0;
    for (ii = 0; ii < fun->args_num; ii++)
    {
        if (ii != fun->idx_in_p && ii != fun->idx_in_global_data)  // skip parameter argument
        {
            fun->args_size_tot += casadi_nnz(fun->casadi_sparsity_in(ii));
        }
    }

    // res
    fun->res_size_tot = 0;
    for (ii = 0; ii < fun->res_num; ii++)
        fun->res_size_tot += casadi_nnz(fun->casadi_sparsity_out(ii));

    acados_size_t size = 0;

    // double pointers
    size += fun->args_num * sizeof(double *);  // args
    size += fun->res_num * sizeof(double *);   // res

    // ints
    size += 2 * fun->args_num * sizeof(int);  // args_size, args_dense
    size += 2 * fun->res_num * sizeof(int);   // res_size, res_dense
    size += fun->int_work_size * sizeof(int);   // int_work

    // doubles
    size += fun->args_size_tot * sizeof(double);  // args
    size += fun->res_size_tot * sizeof(double);   // res
    // float_work
    if (!fun->opts.external_workspace)
    {
        size += fun->float_work_size * sizeof(double);
    }

    size += 8;  // initial align
    size += 8;  // align to double

    make_int_multiple_of(8, &size);

    return size;
}



void external_function_external_param_casadi_assign(external_function_external_param_casadi *fun, void *raw_memory)
{
    int ii;

    // save initial pointer to external memory
    fun->ptr_ext_mem = raw_memory;

    // char pointer for byte advances
    char *c_ptr = raw_memory;

    // double pointers

    // initial align
    align_char_to(8, &c_ptr);

    // args
    assign_and_advance_double_ptrs(fun->args_num, &fun->args, &c_ptr);
    // res
    assign_and_advance_double_ptrs(fun->res_num, &fun->res, &c_ptr);

    // args_size, args_dense
    assign_and_advance_int(fun->args_num, &fun->args_size, &c_ptr);
    assign_and_advance_int(fun->args_num, &fun->args_dense, &c_ptr);
    for (ii = 0; ii < fun->args_num; ii++)
    {
        fun->args_size[ii] = casadi_nnz(fun->casadi_sparsity_in(ii));
        fun->args_dense[ii] = casadi_is_dense(fun->casadi_sparsity_in(ii));
    }
    // res_size, res_dense
    assign_and_advance_int(fun->res_num, &fun->res_size, &c_ptr);
    assign_and_advance_int(fun->res_num, &fun->res_dense, &c_ptr);
    for (ii = 0; ii < fun->res_num; ii++)
    {
        fun->res_size[ii] = casadi_nnz(fun->casadi_sparsity_out(ii));
        fun->res_dense[ii] = casadi_is_dense(fun->casadi_sparsity_out(ii));
    }
    // int_work
    assign_and_advance_int(fun->int_work_size, &fun->int_work, &c_ptr);

    // align to double
    align_char_to(8, &c_ptr);

    // args
    for (ii = 0; ii < fun->args_num; ii++)
    {
        if (ii != fun->idx_in_p && ii != fun->idx_in_global_data)  // skip parameter argument
        {
            assign_and_advance_double(fun->args_size[ii], &fun->args[ii], &c_ptr);
        }
    }

    // res
    for (ii = 0; ii < fun->res_num; ii++)
        assign_and_advance_double(fun->res_size[ii], &fun->res[ii], &c_ptr);
    // float_work
    if (!fun->opts.external_workspace)
    {
        assign_and_advance_double(fun->float_work_size, &fun->float_work, &c_ptr);
    }

    fun->param_mem_is_set = false;
    fun->global_data_ptr_is_set = false;

    assert((char *) raw_memory + external_function_external_param_casadi_calculate_size(fun, &fun->opts) >= c_ptr);

    return;
}


size_t external_function_external_param_casadi_get_external_workspace_requirement(void *self)
{
    external_function_external_param_casadi *fun = self;
    if (fun->opts.external_workspace)
        return fun->float_work_size * sizeof(double);
    else
        return 0;
}

void external_function_external_param_casadi_set_external_workspace(void *self, void *workspace)
{
    external_function_external_param_casadi *fun = self;
    if (fun->opts.external_workspace)
        fun->float_work = workspace;
}


void external_function_external_param_casadi_wrapper(void *self, ext_fun_arg_t *type_in, void **in,
                                            ext_fun_arg_t *type_out, void **out)
{
    // cast into external casadi function
    external_function_external_param_casadi *fun = self;
    int ii;
    int status = 0;

    if (!fun->param_mem_is_set)
    {
        printf("external_function_external_param_casadi_wrapper: attempting to evaluate before parameter memory is set. Exiting.\n");
        exit(1);
    }
    if (!fun->global_data_ptr_is_set && fun->opts.with_global_data)
    {
        printf("external_function_external_param_casadi_wrapper: attempting to evaluate before global data pointer is set. Exiting.\n");
        exit(1);
    }
    // in as args
    for (ii = 0; ii < fun->in_num; ii++)
    {
        // skip parameter arguments
        if (ii != fun->idx_in_p && ii != fun->idx_in_global_data)
            status = d_cvt_ext_fun_arg_to_casadi(type_in[ii], in[ii], (double *) fun->args[ii],
                                    (int *) fun->casadi_sparsity_in(ii), fun->args_dense[ii]);
        if (status)
        {
            printf("\nexternal_function_external_param_casadi_wrapper: Unknown external function argument type %d for input %d\n\n", type_in[ii], ii);
            exit(1);
        }
    }

    // call casadi function
    fun->casadi_fun((const double **) fun->args, fun->res, fun->int_work, fun->float_work, NULL);

    for (ii = 0; ii < fun->out_num; ii++)
    {
        status = d_cvt_casadi_to_ext_fun_arg(type_out[ii], (double *) fun->res[ii], (int *) fun->casadi_sparsity_out(ii),
                                     out[ii], fun->res_dense[ii]);
        if (status)
        {
            printf("\nexternal_function_external_param_casadi_wrapper: Unknown external function argument type %d for output %d\n\n", type_out[ii], ii);
            exit(1);
        }
    }

    return;
}
