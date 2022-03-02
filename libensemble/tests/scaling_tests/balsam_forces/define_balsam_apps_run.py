from balsam.api import ApplicationDefinition, BatchJob


class RemoteForces(ApplicationDefinition):
    site = "jln_theta"
    command_template = (
        "/home/jnavarro"
        + "/libensemble/libensemble/tests/scaling_tests/forces/forces.x"
        + " {{sim_particles}} {{sim_timesteps}} {{seed}} {{kill_rate}}"
        + " > out.txt 2>&1"
    )

    transfers = {
        "result": {
            "required": False,
            "direction": "out",
            "local_path": "forces.stat",
            "description": "Forces stat file",
            "recursive": False,
        }
    }

print("Defined RemoteForces Balsam ApplicationDefinition.")

class LibensembleApp(ApplicationDefinition):
    site = "jln_theta"
    command_template = (
        "/home/jnavarro/.conda/envs/again/bin/python /home/jnavarro"
        + "/libensemble/libensemble/tests/scaling_tests/balsam_forces/run_libe_forces_balsam.py"
        + " > libe_out.txt 2>&1"
    )

    transfers = {
        "input_file": {
            "required": True,
            "direction": "in",
            "local_path": ".",
            "description": "Transfer in of balsam_forces.yaml",
            "recursive": False,
        }
    }

print("Defined LibensembleApp Balsam ApplicationDefinition.")

libe_job = LibensembleApp.submit(
    workdir="libe_workflow/libe_processes",
    transfers={
        "input_file": "theta_dtn:/home/jnavarro/libensemble/libensemble/tests/scaling_tests/balsam_forces/balsam_forces.yaml"
    },
)

print("libEnsemble Job created.")

BatchJob.objects.create(
    site_id=libe_job.site_id,
    num_nodes=4,
    wall_time_min=60,
    job_mode="mpi",
    project="CSC250STMS07",
    queue="debug-flat-quad",
)

print("BatchJob session initialized.")
