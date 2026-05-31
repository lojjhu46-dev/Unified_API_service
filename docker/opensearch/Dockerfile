ARG OPENSEARCH_VERSION=3.6.0
FROM opensearchproject/opensearch:${OPENSEARCH_VERSION}

ARG IK_PLUGIN_URL=https://get.infini.cloud/opensearch/analysis-ik/${OPENSEARCH_VERSION}

RUN /usr/share/opensearch/bin/opensearch-plugin install --batch "${IK_PLUGIN_URL}"
