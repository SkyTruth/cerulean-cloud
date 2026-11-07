from pathlib import Path

import yaml


def test_pulumi_passphrases_come_from_actions_secrets():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/test_and_deploy.yml"
    )
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    expected = {
        "preview-test": ("PULUMI_CONFIG_PASSPHRASE_TEST", 2),
        "deploy-test": ("PULUMI_CONFIG_PASSPHRASE_TEST", 1),
        "deploy-staging": ("PULUMI_CONFIG_PASSPHRASE_STAGING", 1),
        "deploy-production": ("PULUMI_CONFIG_PASSPHRASE_PRODUCTION", 1),
    }
    found = {}
    for job_name, job in jobs.items():
        passphrases = [
            step[section][key]
            for step in job["steps"]
            for section, key in (
                ("env", "PULUMI_CONFIG_PASSPHRASE"),
                ("with", "pulumi-stack-passphrase"),
            )
            if key in step.get(section, {})
        ]
        if passphrases:
            found[job_name] = passphrases
    assert found == {
        job: [f"${{{{ secrets.{secret} }}}}"] * count
        for job, (secret, count) in expected.items()
    }
