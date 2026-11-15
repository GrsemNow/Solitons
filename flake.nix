{
  description = "Solitons";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};

    pythonEnv = pkgs.python3.withPackages (ps: with ps; [
      numpy
      scipy
      matplotlib
    ]);
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [ pythonEnv ];
    };
  };
}
