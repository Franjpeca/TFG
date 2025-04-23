from kedro.config import OmegaConfigLoader

def main():
    loader = OmegaConfigLoader(conf_source="conf")
    try:
        params = loader.get("parameters")  # sin asterisco
        print("✅ Carga directa de 'parameters.yml':")
        print(params)
    except Exception as e:
        print("❌ Error al cargar 'parameters.yml':", str(e))

if __name__ == "__main__":
    main()
