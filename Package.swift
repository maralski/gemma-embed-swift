// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GemmaEmbed",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "GemmaEmbed", targets: ["GemmaEmbed"]),
        .executable(name: "gemma-embed", targets: ["gemma-embed"]),
    ],
    targets: [
        .target(
            name: "GemmaEmbed",
            path: "Sources/GemmaEmbed",
            exclude: ["Model/Architecture.md"],
            resources: [.copy("Shaders")]
        ),
        .executableTarget(
            name: "gemma-embed",
            dependencies: ["GemmaEmbed"],
            path: "Sources/gemma-embed"
        ),
        .testTarget(
            name: "GemmaEmbedTests",
            dependencies: ["GemmaEmbed"],
            path: "Tests/GemmaEmbedTests",
            resources: [.copy("Fixtures")]
        ),
    ]
)
