docker exec -w / \
-it $(docker ps -l --filter "status=running" --format "{{.ID}}") /bin/bash
