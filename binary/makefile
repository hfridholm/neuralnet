# These are the names of the targets
CLEAN_TARGET := clean
HELP_TARGET := help

# These are the flags and command for the clean target
DELETE_FLAGS :=
DELETE_CMD := rm

# This is the compiler and the compile flags you want to use
COMPILER := gcc
COMPILE_FLAGS := -Wall -Werror -g -Og -std=gnu99 -oFast
LINKER_FLAGS := -lm

SOURCE_DIR := ../source
OBJECT_DIR := ../object
BINARY_DIR := ../binary

# These are the source files and object files for the debug library
DEBUG_DIR := $(SOURCE_DIR)/debug

DEBUG_SOURCE_FILES := $(wildcard $(DEBUG_DIR)/*.c)
DEBUG_OBJECT_FILES := $(addprefix $(OBJECT_DIR)/, $(notdir $(DEBUG_SOURCE_FILES:.c=.o)))

# These are the source files and object files for the network library
NETWORK_DIR := $(SOURCE_DIR)/network

NETWORK_SOURCE_FILES := $(wildcard $(NETWORK_DIR)/*.c)
NETWORK_OBJECT_FILES := $(addprefix $(OBJECT_DIR)/, $(notdir $(NETWORK_SOURCE_FILES:.c=.o)))

all: master

%: $(OBJECT_DIR)/%.o $(SOURCE_DIR)/%.c $(DEBUG_OBJECT_FILES) $(DEBUG_SOURCE_FILES) $(NETWORK_OBJECT_FILES) $(NETWORK_SOURCE_FILES)
	$(COMPILER) $(OBJECT_DIR)/$@.o $(DEBUG_OBJECT_FILES) $(NETWORK_OBJECT_FILES) $(LINKER_FLAGS) -o $(BINARY_DIR)/$@

# These are rules for compiling object files out of source files
$(OBJECT_DIR)/%.o: $(SOURCE_DIR)/*/%.c 
	$(COMPILER) $< -c $(COMPILE_FLAGS) -o $@

$(OBJECT_DIR)/%.o: $(SOURCE_DIR)/%.c 
	$(COMPILER) $< -c $(COMPILE_FLAGS) -o $@

# These are the executable programs that you can create
PROGRAMS := $(patsubst %.c, %, $(notdir $(wildcard $(SOURCE_DIR)/*.c)))

# These are the files that should be cleaned
DELETE_FILES := $(OBJECT_DIR)/*.o $(PROGRAMS)

# These are the files that should be kept (not automatically deleted)
.PRECIOUS: $(OBJECT_DIR)/%.o $(PROGRAMS)

$(CLEAN_TARGET):
	$(DELETE_CMD) $(DELETE_FLAGS) $(DELETE_FILES)

$(HELP_TARGET):
	@echo $(PROGRAMS) $(CLEAN_TARGET)

