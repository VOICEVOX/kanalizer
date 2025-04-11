#[test]
fn test_kanalizer() {
    let src = "kanalizer";

    let kanalizer = kanalizer::Kanalizer::new();
    let dst = kanalizer.convert(src, &Default::default()).unwrap();
    assert_eq!(dst, "カナライザー");
}

#[test]
fn test_kanalizer_long() {
    let src = "phosphoribosylaminoimidazolesuccinocarboxamide";

    let kanalizer = kanalizer::Kanalizer::new();
    let unlimited_dst = kanalizer
        .convert(
            src,
            &kanalizer::ConvertOptions {
                max_length: 32.try_into().unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
    let limited_dst = kanalizer
        .convert(
            src,
            &kanalizer::ConvertOptions {
                max_length: 10.try_into().unwrap(),
                error_on_incomplete: false,
                ..Default::default()
            },
        )
        .unwrap();
    assert_ne!(unlimited_dst, limited_dst);
    assert_eq!(limited_dst.chars().count(), 10);
}

#[test]
fn test_kanalizer_long_error() {
    let src = "phosphoribosylaminoimidazolesuccinocarboxamide";

    let kanalizer = kanalizer::Kanalizer::new();
    let limited_dst = kanalizer
        .convert(
            src,
            &kanalizer::ConvertOptions {
                max_length: 10.try_into().unwrap(),
                error_on_incomplete: true,
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(
        limited_dst,
        kanalizer::Error::IncompleteConversion { .. }
    ));
}

#[test]
fn test_validate_empty() {
    let src = "";

    let kanalizer = kanalizer::Kanalizer::new();
    let err = kanalizer.convert(src, &Default::default()).unwrap_err();
    assert_eq!(err, kanalizer::Error::EmptyInput);
}

#[rstest::rstest]
#[test]
#[case("あ")]
#[case("A")]
fn test_validate_invalid_chars(#[case] char: &str) {
    let kanalizer = kanalizer::Kanalizer::new();
    let err = kanalizer.convert(char, &Default::default()).unwrap_err();
    assert_eq!(
        err,
        kanalizer::Error::InvalidChars {
            chars: vec![char.chars().next().unwrap()]
        }
    );
}
