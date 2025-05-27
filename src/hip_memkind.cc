#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>

#include "mpi.h"

#define MAX_TESTS     16
#define MAX_SCENARIOS 5

static char info_val_empty[] = "";
static char info_val_system[] = "system";
static char info_val_mpi[] = "mpi";
static char info_val_system_mpi[] = "system,mpi";
static char info_val_system_mpi_rocm[] = "system,mpi,rocm";
static char info_val_system_mpi_rocmdevice[] = "system,mpi,rocm:device";

static char scenario_0[] = "no memkind argument provided";
static char scenario_1[] = "memkind = system,mpi";
static char scenario_2[] = "memkind = system,mpi,rocm";
static char scenario_3[] = "memkind = system,mpi,rocm:device";
static char scenario_4[] = "memkind = system,mpi,rocm,nonsense";

static char* set_expected_values (int scenario, bool expected_found[MAX_TESTS], char *expected_value[MAX_TESTS])
{
  if (scenario == 0) {
    for (int i=0; i<MAX_TESTS; i++)
      expected_found[i] = false;
  } else {
    for (int i=0; i<MAX_TESTS; i++)
      expected_found[i] = true;
  }

  if (scenario == 0) {
    /* no memkind value set at mpirun */
    for (int i=0; i<MAX_TESTS;i++)
      expected_value[i] = info_val_empty;

    return scenario_0;
  } else if (scenario == 1) {
    /* memkind = system,mpi */
    for (int i=0; i<MAX_TESTS;i++)
      expected_value[i] = info_val_system_mpi;

    /* Adjust the tests that return different info value */
    expected_value[5] = info_val_mpi;
    expected_value[6] = info_val_mpi;
    expected_value[13] = info_val_mpi;

    return scenario_1;
  } else if (scenario == 2 || scenario == 4) {
    /* memkind = system,mpi,rocm */
    for (int i=0; i<MAX_TESTS;i++)
      expected_value[i] = info_val_system_mpi_rocm;

    /* Adjust the tests that return different info value */
    expected_value[5] = info_val_mpi;
    expected_value[6] = info_val_mpi;
    expected_value[13] = info_val_mpi;

    if (scenario == 2) return scenario_2;
  } else if (scenario == 3) {
    /* memkind = system,mpi,rocm:device */
    for (int i=0; i<MAX_TESTS;i++)
      expected_value[i] = info_val_system_mpi_rocmdevice;

    /* Adjust the tests that return different info value */
    expected_value[5] = info_val_mpi;
    expected_value[6] = info_val_mpi;
    expected_value[13] = info_val_mpi;

    return scenario_3;
  }

  return scenario_4;  
}

static int check_comm_memkind_info (MPI_Comm comm, const char *comm_name, const char *info_name,
				    bool expected_found, const char *expected_val)
{
  MPI_Info info;
  int len = 0, flag = 0;
  int rank;
  int localret = 0, globalret = 0;
  char *val=NULL, *valptr=NULL;
  MPI_Comm_rank (comm, &rank);
  
  MPI_Comm_get_info(comm, &info);
  MPI_Info_get_string(info, info_name, &len, NULL, &flag);

  if (flag) {
    val = valptr = (char *) malloc(len);
    if (NULL == val) return 1;
    
    MPI_Info_get_string(info, info_name, &len, val, &flag);
    localret = strncmp (val, expected_val, len);
  } else {
    if (expected_found) localret = 1;    
  }

  MPI_Allreduce (&localret, &globalret, 1, MPI_INT, MPI_MAX, comm);
  if (rank == 0) {
    printf("Comm: %-45.45s \t %s\n", comm_name, globalret == 0 ? "SUCCESS" : "FAILURE");
    if (globalret != 0) printf("expected %s got %s\n", expected_val, val);
  }

  free(valptr);
  return globalret;
}

static int check_win_memkind_info (MPI_Win win, MPI_Comm comm, const char *win_name,
				    bool expected_found, const char *expected_val)
{
  MPI_Info info;
  int len = 0, flag = 0;
  int rank;
  int localret = 0, globalret = 0;
  char *val=NULL, *valptr=NULL;
  MPI_Comm_rank (comm, &rank);
  
  MPI_Win_get_info(win, &info);
  MPI_Info_get_string(info, "mpi_memory_alloc_kinds",
		      &len, NULL, &flag);

  if (flag) {
    val = valptr = (char *) malloc(len);
    if (NULL == val) return 1;
    
    MPI_Info_get_string(info, "mpi_memory_alloc_kinds",	&len, val, &flag);
    localret = strncmp (val, expected_val, len);
  } else {
    if (expected_found) localret = 1;
  }

  MPI_Allreduce (&localret, &globalret, 1, MPI_INT, MPI_MAX, comm);
  if (rank == 0) {
    printf("Win: %-45.45s \t %s\n", win_name, globalret == 0 ? "SUCCESS" : "FAILURE");
    if (globalret != 0) printf("expected %s got %s\n", expected_val, val);
  }

  free(valptr);
  return globalret;
}

static int check_file_memkind_info (MPI_File file, MPI_Comm comm, const char *file_name,
				    bool expected_found, const char *expected_val)
{
  MPI_Info info;
  int len = 0, flag = 0;
  int rank;
  int localret = 0, globalret = 0;
  char *val=NULL, *valptr=NULL;
  MPI_Comm_rank (comm, &rank);
  
  MPI_File_get_info(file, &info);
  MPI_Info_get_string(info, "mpi_memory_alloc_kinds", &len, NULL, &flag);

  if (flag) {
    val = valptr = (char *) malloc(len);
    if (NULL == val) return 1;
    
    MPI_Info_get_string(info, "mpi_memory_alloc_kinds",	&len, val, &flag);
    localret = strncmp (val, expected_val, len);
  } else {
    if (expected_found) localret = 1;
  }

  MPI_Allreduce (&localret, &globalret, 1, MPI_INT, MPI_MAX, comm);
  if (rank == 0) {
    printf("File: %-45.45s \t %s\n", file_name, globalret == 0 ? "SUCCESS" : "FAILURE");
    if (globalret != 0) printf("expected %s got %s\n", expected_val, val);
  }

  free(valptr);
  return globalret;
}

#ifdef HIP_MPITEST_SESSIONS
static int check_session_memkind_info (MPI_Session session, const char *session_name,
				       bool expected_found, const char *expected_val)
{
  MPI_Info info;
  int len = 0, flag = 0;
  int ret = 0;
  char *val=NULL, *valptr=NULL;

  MPI_Session_get_info(session, &info);
  MPI_Info_get_string(info, "mpi_memory_alloc_kinds",
		      &len, NULL, &flag);

  if (flag) {
    val = valptr = (char *) malloc(len);
    if (NULL == val) return 1;

    MPI_Info_get_string(info, "mpi_memory_alloc_kinds",	&len, val, &flag);
    ret = strncmp (val, expected_val, len);
  } else {
    if (expected_found) ret = 1;
  }

  printf("Session: %-45.45s \t %s\n", session_name, ret == 0 ? "SUCCESS" : "FAILURE");
  if (ret != 0) printf("expected %s got %s\n", expected_val, val);

  free(valptr);
  return ret;
}

int main (int argc, char **argv)
{
  int rank, size;
  int ret = 0;
  MPI_Session session;
  MPI_Group wgroup;

  MPI_Info info;
  MPI_Info_create (&info);
  char key[] = "mpi_memory_alloc_kinds";
  char value[] = "rocm,system,mpi";
  MPI_Info_set (info, key, value);

  MPI_Info info_assert;
  MPI_Info_create (&info_assert);
  char assert_key[] = "mpi_assert_memory_alloc_kinds";
  char assert_value[] = "mpi";
  MPI_Info_set (info_assert, assert_key, assert_value);

  MPI_Session_init(info, MPI_ERRORS_ARE_FATAL, &session);
  MPI_Info_free(&info);

  ret += check_session_memkind_info(session, "Session init with info set",
				    true, info_val_system_mpi_rocm);

  MPI_Group_from_session_pset(session, "mpi://WORLD" , &wgroup);

  MPI_Comm comm;
  MPI_Comm_create_from_group(wgroup, "mem-alloc-kind-example",
                             MPI_INFO_NULL, MPI_ERRORS_ABORT, &comm);
  ret += check_comm_memkind_info (comm, "Comm_create_from_group with INFO_NULL",
				  "mpi_memory_alloc_kinds", true, info_val_system_mpi_rocm);
  MPI_Comm_free (&comm);

  MPI_Comm_create_from_group(wgroup, "mem-alloc-kind-example-assert",
                             info_assert, MPI_ERRORS_ABORT, &comm);

  ret += check_comm_memkind_info (comm, "Comm_create_from_group with info_assert",
				  "mpi_memory_alloc_kinds", true, info_val_mpi);
  MPI_Comm_free (&comm);

  MPI_Session_finalize (&session);
  return ret;
}
#else
int main (int argc, char **argv)
{
  int rank, size;
  int scenario=0;
  char *expected_values[MAX_TESTS];
  bool expected_found[MAX_TESTS];
  int ret = 0;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  if (argc > 1) {
    scenario = atoi(argv[1]);
  }

  char *scenario_name = set_expected_values(scenario, expected_found, expected_values);
  if (rank == 0) {
    printf("Testing memkind scenario %d : %s\n", scenario, scenario_name);
    char *memkind_requested = getenv ("OMPI_MCA_mpi_memory_alloc_kinds");
    if (memkind_requested) {
      printf("OMPI_MCA_mpi_memory_alloc_kinds = %s\n", memkind_requested);
    }
    printf("=============================================================\n");
  }

  MPI_Info info;
  MPI_Info_create (&info);
  char key[] = "mpi_memory_alloc_kinds";
  char value[] = "rocm,system,mpi";
  MPI_Info_set (info, key, value);

  MPI_Info info_assert;
  MPI_Info_create (&info_assert);
  char assert_key[] = "mpi_assert_memory_alloc_kinds";
  char assert_value[] = "mpi";
  MPI_Info_set (info_assert, assert_key, assert_value);

  MPI_Info info_invalid;
  MPI_Info_create (&info_invalid);
  char invalid_key[] = "mpi_assert_memory_alloc_kinds";
  char invalid_value[] = "mpi:nonsense";
  MPI_Info_set (info_invalid, invalid_key, invalid_value);

  
#ifdef VERBOSE  
  printf("Hello World from rank %d size %d\n", rank, size);
#endif

  /* test 0 */
  ret += check_comm_memkind_info (MPI_COMM_WORLD, "MPI_COMM_WORLD", "mpi_memory_alloc_kinds",
				  expected_found[0], expected_values[0]);

  /* test 1 */
  ret += check_comm_memkind_info (MPI_COMM_SELF, "MPI_COMM_SELF", "mpi_memory_alloc_kinds",
				  expected_found[1], expected_values[1]);

  /* test 2 */
  MPI_Comm comm_dup;
  MPI_Comm_dup (MPI_COMM_WORLD, &comm_dup);
  ret += check_comm_memkind_info (comm_dup, "MPI_Comm_dup", "mpi_memory_alloc_kinds",
				  expected_found[2], expected_values[2]);

  /* test 3 */
  MPI_Comm_set_info (comm_dup, info);
  ret += check_comm_memkind_info (comm_dup, "MPI_Comm_dup after Comm_set_info", "mpi_memory_alloc_kinds",
				  expected_found[3], expected_values[3]);
  MPI_Comm_free (&comm_dup);

  /* test 4 */
  MPI_Comm_dup_with_info (MPI_COMM_WORLD, MPI_INFO_NULL, &comm_dup);
  ret += check_comm_memkind_info (comm_dup, "MPI_Comm_dup_with_info & INFO_NULL", "mpi_memory_alloc_kinds",
				  expected_found[4], expected_values[4]);
  MPI_Comm_free (&comm_dup);

  /* test 5 */
  MPI_Comm_dup_with_info (MPI_COMM_WORLD, info_assert, &comm_dup);
  ret +=  check_comm_memkind_info (comm_dup, "MPI_Comm_dup_with_info & info_assert", "mpi_memory_alloc_kinds",
				   expected_found[5], expected_values[5]);

  /* test 6 */
  ret +=  check_comm_memkind_info (comm_dup, "MPI_Comm_dup_with_info & info_assert", "mpi_assert_memory_alloc_kinds",
				   expected_found[6], expected_values[6]);

  /* test 7 */
  MPI_Comm comm_dup2;
  MPI_Comm_dup (comm_dup, &comm_dup2);
  ret += check_comm_memkind_info (comm_dup2, "MPI_Comm_dup of comm_dup'ed with info_assert", "mpi_memory_alloc_kinds",
				  expected_found[7], expected_values[7]);
  MPI_Comm_free (&comm_dup);
  MPI_Comm_free (&comm_dup2);

  /* test 8 */
  MPI_Comm_dup_with_info (MPI_COMM_WORLD, info_invalid, &comm_dup);
  ret += check_comm_memkind_info (comm_dup, "MPI_Comm_dup_with_info & info_invalid", "mpi_memory_alloc_kinds",
				  expected_found[8], expected_values[8]);
  MPI_Comm_free (&comm_dup);

  /* test 9 */
  MPI_Comm comm_create;
  MPI_Group group;
  MPI_Comm_group (MPI_COMM_WORLD, &group);
  MPI_Comm_create (MPI_COMM_WORLD, group, &comm_create);
  ret += check_comm_memkind_info (comm_create, "MPI_Comm_create", "mpi_memory_alloc_kinds",
				  expected_found[9], expected_values[9]);
  MPI_Group_free (&group);
  MPI_Comm_free (&comm_create);  

  /* test 10 */
  MPI_Comm inter_comm;
  MPI_Intercomm_create (MPI_COMM_SELF, 0, MPI_COMM_WORLD, ((rank + 1)%2), 17, &inter_comm);
  ret += check_comm_memkind_info (inter_comm, "MPI_Intercomm_create", "mpi_memory_alloc_kinds",
				  expected_found[10], expected_values[10]);

  /* test 11 */
  MPI_Comm inter_merge;
  MPI_Intercomm_merge (inter_comm, rank, &inter_merge);
  ret += check_comm_memkind_info (inter_merge, "MPI_Intercomm_merge", "mpi_memory_alloc_kinds",
				  expected_found[11], expected_values[11]);
  MPI_Comm_free (&inter_comm);
  MPI_Comm_free (&inter_merge);
  
  /* test 12 */
  MPI_File file;
  MPI_File_open (MPI_COMM_WORLD, "testfile1.out", MPI_MODE_CREATE|MPI_MODE_WRONLY,
		 MPI_INFO_NULL, &file);
  ret += check_file_memkind_info (file, MPI_COMM_WORLD, "MPI_File_open with INFO_NULL",
				  expected_found[12], expected_values[12]);
  MPI_File_close (&file);
  unlink ("testfile1.out");

  /* test 13 */
  MPI_File_open (MPI_COMM_WORLD, "testfile1.out", MPI_MODE_CREATE|MPI_MODE_WRONLY,
		 info_assert, &file);
  ret += check_file_memkind_info (file, MPI_COMM_WORLD, "MPI_File_open with info_assert",
				  expected_found[13], expected_values[13]);
  MPI_File_close (&file);
  unlink ("testfile1.out");

  /* test 14 */
  MPI_File_open (MPI_COMM_WORLD, "testfile1.out", MPI_MODE_CREATE|MPI_MODE_WRONLY,
		 info_invalid, &file);
  ret += check_file_memkind_info (file, MPI_COMM_WORLD, "MPI_File_open with info_invalid",
				  expected_found[14], expected_values[14]);
  MPI_File_close (&file);
  unlink ("testfile1.out");
  
  /* test 15 */
  MPI_Win win;
  int buffer[16];
  MPI_Win_create (buffer, 16*sizeof(int), sizeof(int), MPI_INFO_NULL,
		  MPI_COMM_WORLD, &win);
  ret += check_win_memkind_info (win, MPI_COMM_WORLD, "MPI_Win_create",
				 expected_found[15], expected_values[15]);
  MPI_Win_free (&win);

  if (rank == 0) printf("\n");
  
  MPI_Info_free (&info);
  MPI_Info_free (&info_assert);
  MPI_Info_free (&info_invalid);
  MPI_Finalize();

  return ret;
}
#endif
