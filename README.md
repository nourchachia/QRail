🚄 QRail: An AI-Powered Operational Memory

QRail builds a searchable memory of historical rail incidents and their resolutions. When a new disruption occurs, the system:

1. Encodes the incident as multi-dimensional embeddings (spatial + temporal + contextual)
2. Retrieves the most similar historical cases using Qdrant
3. Recommends resolution strategies based on what worked before
4. Explains recommendations with evidence from historical cases
5. Learns by storing outcomes and operator feedback

