<?php

$context["active_ui"] = CoreExtension::getAttribute($this->env, $this->source, CoreExtension::getAttribute($this->env, $this->source, (isset($context["app"]) || array_key_exists("app", $context) ? $context["app"] : (function () {
    throw new RuntimeError('Variable "app" does not exist.', 84, $this->source);
})()), "request", [], "any", false, false, false, 84), "get", ["ui", "swagger_ui"], "method", false, false, false, 84);
