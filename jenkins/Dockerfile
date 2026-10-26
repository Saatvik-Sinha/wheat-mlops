# Local Jenkins controller for Pipeline B, with uv and Python preinstalled.
# Build and run with docker compose (see jenkins/README.md).
#
# The LTS line tag is used (not a fixed version) so the plugins below, which are installed
# at their latest versions, are always compatible with the Jenkins core.
FROM jenkins/jenkins:lts-jdk21

USER root

# uv/uvx pinned to the version the team develops with.
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /usr/local/bin/

# Python is installed by uv into a shared location so the jenkins user can use it.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed
RUN uv python install 3.14 && chmod -R a+rX /opt/uv-python

USER jenkins

# Keep uv's download cache in the persistent Jenkins home volume.
ENV UV_CACHE_DIR=/var/jenkins_home/.cache/uv

COPY --chown=jenkins:jenkins plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt
